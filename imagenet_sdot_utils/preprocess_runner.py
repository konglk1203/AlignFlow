import os
import torch
from tqdm import tqdm
import concurrent.futures
from compute_sdot import compute_sdot_class
from generate_sdot_dataset import generate_sdot_dataset_class
import argparse

NUM_CLASSES_IMAGENET = int(os.environ["NUM_CLASSES_IMAGENET"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, help="Task in [compute_sdot, generate_sdot_dataset]"
    )
    args = parser.parse_args()

    task_dict = {
        "compute_sdot": compute_sdot_class,
        "generate_sdot_dataset": generate_sdot_dataset_class,
    }

    func_single_gpu = task_dict[args.task]

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    completed_count = 0
    gpu_available = [True] * num_gpus

    with tqdm(
        total=NUM_CLASSES_IMAGENET, desc="Total process", position=num_gpus
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = set()

            # Initial submission of tasks
            for gpu_id in range(min(num_gpus, NUM_CLASSES_IMAGENET)):
                class_id = gpu_id
                futures.add(executor.submit(func_single_gpu, class_id, gpu_id))
                gpu_available[gpu_id] = False

            class_id = num_gpus
            while completed_count < NUM_CLASSES_IMAGENET:
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    try:
                        completed_class_id, used_gpu_id = future.result()

                        # Mark GPU as available again
                        gpu_available[used_gpu_id] = True
                        completed_count += 1
                        pbar.update(1)

                        # Submit new task if there are classes left
                        if class_id < NUM_CLASSES_IMAGENET:
                            futures.add(
                                executor.submit(func_single_gpu, class_id, used_gpu_id)
                            )
                            gpu_available[used_gpu_id] = False
                            class_id += 1

                    except Exception as e:
                        pbar.write(f"An error occurred: {e}")
