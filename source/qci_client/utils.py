"""Utilities for qci-client."""

import concurrent.futures
from typing import Callable, Generator, Optional, Union


def _buffered_generator(
    *,
    buffer_length: int,
    task: Callable,
    args_kwargs_generator: Generator,
    timeout: Optional[Union[int, float]] = 0.01,
) -> Generator:
    """
    Generator that maps concurrent tasks over iterables using a buffer to limit memory
    pressure.

    :param buffer_length: Length of FIFO buffer, often equal to a thread/process count.
    :param task: A future-returning function to execute concurrently.
    :param args_kwargs_generator: Generator of (args, kwargs) tuple to pass to each
        task, where args is a tuple and kwargs is a dictionary.
    :param timeout: Timeout to wait for future results to resolve before looping, [s].

    :return: Generator using a concurrently-filled FIFO buffer.

    Notes:
    - Caller is expected to enforce any resource limits affecting choice of arguments.
    - Caller is expected to throw any Exception during yield back to generator to cancel
      any pending tasks and close generator gracefully (via return raising StopIteration
      in caller).
    """

    def async_generator() -> Generator:  # pylint disable=too-many-branches
        """Generates results of a task computed concurrently via a FIFO buffer."""

        pending_results = []  # FIFO buffer.
        result_ready = False
        has_input = True

        # Outer loop is cancelled from caller by caller throwing an exception back to
        # generator from within the yield.
        while True:
            if pending_results:
                try:
                    # Check if oldest task is done. Yielding via timeout.
                    result = pending_results[0].result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    # Let loop iterate before another result check.
                    pass
                except Exception:  # Includes CancelledError.
                    # Cancel any/all tasks before bubbling up exception.
                    for pending_result in pending_results:
                        pending_result.cancel()
                    raise
                else:
                    # Remove oldest task that resolved of front of list.
                    pending_results.pop(0)
                    # Yield result after refilling FIFO buffer.
                    result_ready = True

            # Submit tasks into FIFO buffer for remaining input until max_worker
            # tasks have been added.
            while has_input and len(pending_results) < buffer_length:
                try:
                    args_generated, kwargs_generated = next(args_kwargs_generator)
                    pending_results.append(task(*args_generated, **kwargs_generated))
                except StopIteration:
                    # Generated input ran out.
                    has_input = False

            # Yield generated result, if ready.
            if result_ready:
                try:
                    yield result
                except Exception:  # pylint: disable=broad-exception-caught
                    # Cancel any/all tasks before returning.
                    for pending_result in pending_results:
                        pending_result.cancel()
                    return

                result_ready = False

            # If there are no pending results, then generator is finished.
            if not pending_results:
                return

    return async_generator()
