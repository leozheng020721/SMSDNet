"""Small native-process checks; no model or production imports are needed."""

from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
import os
import sys
import unittest

from recover_seo_nir_pan_isolated import IsolatedImageExecutor


def _payload(value, *, multiplier=1, offset=0):
    return {
        "pid": os.getpid(),
        "value": value * multiplier + offset,
        "production_imported": "recover_seo_nir_pan" in sys.modules,
        "torch_imported": "torch" in sys.modules,
    }


def _raise_value_error(message):
    raise ValueError(message)


def _exit_without_result():
    os._exit(7)


class IsolatedImageExecutorTests(unittest.TestCase):
    def test_three_jobs_have_distinct_child_pids_and_preserve_kwargs(self):
        parent_pid = os.getpid()
        with IsolatedImageExecutor(max_workers=1) as executor:
            expected = {
                executor.submit(_payload, index, multiplier=3, offset=2): index * 3 + 2
                for index in range(3)
            }
            results = []
            for future in as_completed(expected, timeout=30):
                result = future.result()
                self.assertEqual(result["value"], expected[future])
                self.assertNotEqual(result["pid"], parent_pid)
                self.assertFalse(result["production_imported"])
                self.assertFalse(result["torch_imported"])
                results.append(result)
        self.assertEqual(len({result["pid"] for result in results}), 3)

    def test_worker_exception_reaches_future_and_next_job_runs(self):
        with IsolatedImageExecutor() as executor:
            failed = executor.submit(_raise_value_error, "synthetic worker failure")
            following = executor.submit(_payload, 8, multiplier=2)
            with self.assertRaisesRegex(ValueError, "synthetic worker failure"):
                failed.result(timeout=30)
            self.assertEqual(following.result(timeout=30)["value"], 16)
            self.assertIsInstance(failed.exception(), ValueError)

    def test_hard_worker_exit_reaches_future_and_next_job_runs(self):
        with IsolatedImageExecutor() as executor:
            failed = executor.submit(_exit_without_result)
            following = executor.submit(_payload, 11, offset=1)
            with self.assertRaises(BrokenProcessPool):
                failed.result(timeout=30)
            result = following.result(timeout=30)
            self.assertEqual(result["value"], 12)
            self.assertNotEqual(result["pid"], os.getpid())
            self.assertIsInstance(failed.exception(), BrokenProcessPool)

    def test_context_exit_waits_for_queued_results(self):
        with IsolatedImageExecutor() as executor:
            futures = [executor.submit(_payload, index) for index in range(2)]
        self.assertTrue(all(future.done() for future in futures))
        self.assertEqual([future.result()["value"] for future in futures], [0, 1])

    def test_context_preserves_caller_exception(self):
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with IsolatedImageExecutor():
                raise RuntimeError("caller failure")

    def test_rejects_multiple_workers(self):
        for workers in (2, 3):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers 1"):
                    IsolatedImageExecutor(max_workers=workers)

    def test_cannot_submit_after_shutdown(self):
        with IsolatedImageExecutor() as executor:
            pass
        with self.assertRaises(RuntimeError):
            executor.submit(_payload, 1)


if __name__ == "__main__":
    unittest.main()
