# Copyright 2025 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Where to dump the prediction results.",
    )  # noqa

    # model
    args.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    args.add_argument("--trust_remote_code", action="store_true")

    args.add_argument(
        "--attn",
        type=str,
        default="dense",
    )

    args.add_argument(
        "--cfg",
        type=str,
        default="",
        help=(
            "Additional kwargs for attention implementation, specified as "
            "'key1=value1,key2=value2', default to ''"
        ),
    )

    args.add_argument("--run_benchmark", action="store_true")
    args.add_argument(
        "--num_exp",
        type=int,
        default=10,
        help="Number of experiments to average over",
    )
    args.add_argument("--context_length", type=int, default=128000)

    return args.parse_args()
