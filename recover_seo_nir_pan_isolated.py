"""Resume PAN recovery with a fresh worker process for every image.

Only process lifetime changes. The production pipeline, configuration hash,
quality thresholds, and output/resume behavior stay in the production module.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import freeze_support, get_context


def _run_in_fresh_process(function, args, kwargs):
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(function, *args, **kwargs).result()


class IsolatedImageExecutor:
    """Queue images serially and release all native allocations after each one."""

    def __init__(self, max_workers=1):
        if max_workers != 1:
            raise ValueError("Isolated recovery requires --workers 1.")
        self._queue = ThreadPoolExecutor(max_workers=1)

    def submit(self, function, *args, **kwargs):
        return self._queue.submit(_run_in_fresh_process, function, args, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._queue.shutdown(wait=True, cancel_futures=exc_type is not None)
        return False


def main():
    import recover_seo_nir_pan as production

    production.ProcessPoolExecutor = IsolatedImageExecutor
    return production.main()


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
