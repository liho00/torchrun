# common_distributed.py

import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import traceback
import types

import torch
import torch.cuda.nccl
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Most tests operate with this worldsize
DEFAULT_WORLD_SIZE = 4

# [How does MultiProcess work?]
# Each MultiProcess instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 4. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcess._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.

class MultiProcess():
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    ERROR_EXIT_CODE = 10

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name: str = "") -> None:
        # pass
        self.skip_return_code_checks = []  # type: ignore[var-annotated]
        self.processes = []  # type: ignore[var-annotated]
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        # pid to pipe consisting of error message from process.
        self.pid_to_pipe = {}  # type: ignore[var-annotated]

    def destroy(self) -> None:
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _start_processes(self, method_name) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process

        self.processes = []

        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, method_name, self.file_name, child_conn),
            )
            process.start()
            logger.info("Started process %s with pid %s", rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    @classmethod
    def _run(cls, rank: int, method_name: str, file_name: str, parent_pipe, **kwargs) -> None:
        self = cls(method_name)
        self.rank = rank
        self.file_name = file_name
        self.run(method_name, parent_pipe)

    def run(self, method_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        if sys.platform != "win32" and sys.platform != "darwin":
            # Register signal handler to dump stack traces on FATALs.
            # Windows and MacOS do not support the signal handlers.
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        try:
            getattr(self, method_name)()
        except Exception as e:
            logger.error(
                "Caught exception: \n%s exiting "
                "process %s with exit code: %s",
                traceback.format_exc(), self.rank, MultiProcess.ERROR_EXIT_CODE
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcess.ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

    @property
    def is_master(self) -> bool:
        return self.rank == 0



