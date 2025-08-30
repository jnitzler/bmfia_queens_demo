import logging
from time import time

import numpy as np
import tqdm
from dask.distributed import as_completed
from queens.schedulers.local import Local
from queens.utils.logger_settings import log_init_args
from queens.utils.printing import get_str_table

_logger = logging.getLogger(__name__)


class LocalWFeatures(Local):
    """Local scheduler class for QUEENS with feature output."""

    @log_init_args
    def __init__(
        self,
        experiment_name,
        num_jobs=1,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    ):
        """Initialize local scheduler.

        Args:
            experiment_name (str): name of the current experiment
            num_jobs (int, opt): Maximum number of parallel jobs
            num_procs (int, opt): number of processors per job
            restart_workers (bool): If true, restart workers after each finished job. Try setting it
                                    to true in case you are experiencing memory-leakage warnings.
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        super().__init__(
            experiment_name=experiment_name,
            num_jobs=num_jobs,
            num_procs=num_procs,
            restart_workers=restart_workers,
            verbose=verbose,
        )

    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """
        if self.restart_workers:
            # This is necessary, because the subprocess in the driver does not get killed
            # sometimes when the worker is restarted.
            def run_driver(*args, **kwargs):
                time.sleep(5)
                return driver.run(*args, **kwargs)

        else:
            run_driver = driver.run

        if job_ids is None:
            job_ids = self.get_job_ids(len(samples))

        futures = self.client.map(
            run_driver,
            samples,
            job_ids,
            pure=False,
            num_procs=self.num_procs,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )

        # The theoretical number of sequential jobs
        num_sequential_jobs = int(np.ceil(len(samples) / self.num_jobs))

        results = {future.key: None for future in futures}
        with tqdm.tqdm(total=len(futures)) as progressbar:
            for future in as_completed(futures):
                results[future.key] = future.result()
                progressbar.update(1)
                if self.restart_workers:
                    worker = list(self.client.who_has(future).values())[0]
                    self.restart_worker(worker)

            if self.verbose:
                elapsed_time = progressbar.format_dict["elapsed"]
                averaged_time_per_job = elapsed_time / num_sequential_jobs

                run_time_dict = {
                    "number of jobs": len(samples),
                    "number of parallel jobs": self.num_jobs,
                    "number of procs": self.num_procs,
                    "total elapsed time": f"{elapsed_time:.3e}s",
                    "average time per parallel job": f"{averaged_time_per_job:.3e}s",
                }
                _logger.info(
                    get_str_table(
                        f"Batch summary for jobs {min(job_ids)} - {max(job_ids)}",
                        run_time_dict,
                    )
                )

        result_dict = {
            "result": [],
            "gradient": [],
            "features": [],
        }  # this is different to Local Scheduler
        for result in results.values():
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict["result"].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict["gradient"].append(result[1])
            result_dict["features"].append(
                result[2]
            )  # this is different to Local Scheduler
        result_dict["result"] = np.array(result_dict["result"])
        result_dict["gradient"] = np.array(result_dict["gradient"])
        result_dict["features"] = np.array(
            result_dict["features"]
        )  # this is different to Local Scheduler
        return result_dict
