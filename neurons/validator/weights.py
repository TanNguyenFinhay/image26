# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
from typing import List

import torch
from loguru import logger

import bittensor as bt

from neurons.validator.signed_requests import SignedRequests
from neurons.validator.utils import get_validator_spec_version


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Utils for weights setting on chain.


def post_weights(
    hotkey, api_url: str, hotkeys: List[str], raw_weights: torch.Tensor
):
    response = SignedRequests(hotkey=hotkey).post(
        f"{api_url}/validator/weights",
        json={
            "weights": {
                hotkey: moving_average.item()
                for hotkey, moving_average in zip(hotkeys, raw_weights)
            }
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    return response


def set_weights(val):
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(
        val.moving_average_scores, p=1, dim=0
    )

    try:
        response = post_weights(
            val.wallet.hotkey,
            val.api_url,
            val.hotkeys,
            raw_weights,
        )
        if response.status_code != 200:
            logger.info("Error logging weights to the Weights API")
        else:
            logger.info("Successfully logged weights to the Weights API")
    except Exception:
        logger.info("Error logging weights to the Weights API")

    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=val.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=val.config.netuid,
        subtensor=val.subtensor,
        metagraph=val.metagraph,
    )
    logger.info("processed_weights", processed_weights)
    logger.info("processed_weight_uids", processed_weight_uids)
    from neurons.validator import validator
    # Set the weights on chain via our subtensor connection.
    val.subtensor.set_weights(
        wallet=val.wallet,
        netuid=val.config.netuid,
        uids=processed_weight_uids,
        weights=processed_weights,
        wait_for_finalization=False,
        version_key=get_validator_spec_version(),
    )
