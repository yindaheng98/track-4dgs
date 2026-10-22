"""Shared arguments plus optional repeated groups."""

from abc import abstractmethod

from .dataset import AbstractDataset


class AbstractGroupArgsDataset(AbstractDataset):
    """Dataset built from shared arguments and, optionally, repeated groups.

    :meth:`parse_known_common_argv` parses the flags shared by every sample.
    :meth:`parse_known_group_chunk` parses one repeated group.
    :attr:`arg_group_starters` lists the option strings that begin a group.
    An empty :attr:`arg_group_starters` means this dataset has no groups.
    :meth:`from_argv` parses each group, then the shared flags left over.
    :meth:`from_args` receives the shared namespace and one namespace per group.
    """

    arg_group_starters: tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def parse_known_group_chunk(cls, argv: list[str] = []):
        """Parse one group's argv."""
        raise NotImplementedError

    @classmethod
    def parse_known_group_argv(cls, argv: list[str] = []):
        """Split ``argv`` into groups and parse each one.

        Each chunk starts at one :attr:`arg_group_starters` token and stops before the next.
        The last chunk runs through the end of ``argv``. Tokens before the first starter
        are returned with the unrecognized arguments from every group.
        """
        starters = set(cls.arg_group_starters)
        starts = [index for index, token in enumerate(argv) if token in starters]
        groups = []
        rest = list(argv[:starts[0]]) if starts else list(argv)
        for begin, end in zip(starts, starts[1:] + [len(argv)]):
            group, extra = cls.parse_known_group_chunk(argv[begin:end])
            groups.append(group)
            rest.extend(extra)
        return groups, rest

    @classmethod
    @abstractmethod
    def parse_known_common_argv(cls, argv: list[str] = []):
        """Parse the arguments shared by every sample."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_args(cls, common_args, *group_args) -> 'AbstractGroupArgsDataset':
        """Build this dataset from the shared namespace and one namespace per group."""
        raise NotImplementedError

    @classmethod
    def from_argv(cls, argv: list[str] = []) -> tuple['AbstractGroupArgsDataset', list[str]]:
        """Parse groups, then the shared arguments left over, and call :meth:`from_args`.

        The second value is the unrecognized remainder of ``argv``.
        """
        groups, unknown = cls.parse_known_group_argv(argv)
        common_args, unknown = cls.parse_known_common_argv(unknown)
        return cls.from_args(common_args, *groups), unknown
