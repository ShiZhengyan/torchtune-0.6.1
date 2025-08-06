# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Union

from torchtune.utils import get_logger

log = get_logger("DEBUG")


def update_tokenizer_json_added_tokens(
    tokenizer,
    tokenizer_json_path: Union[str, Path],
) -> None:
    """
    Update only the added_tokens section in tokenizer.json with new tokens.
    
    Args:
        tokenizer: The extended torchtune tokenizer instance
        tokenizer_json_path: Path to the tokenizer.json file to update
    """
    tokenizer_json_path = Path(tokenizer_json_path)
    
    if not tokenizer_json_path.exists():
        log.warning(f"tokenizer.json not found at {tokenizer_json_path}")
        return
    
    # Load original tokenizer.json
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    
    # Ensure added_tokens section exists
    if "added_tokens" not in tokenizer_data:
        tokenizer_data["added_tokens"] = []
    
    # Create a set of existing token IDs to avoid duplicates
    existing_token_ids = set()
    for token_entry in tokenizer_data["added_tokens"]:
        existing_token_ids.add(token_entry.get("id"))
    
    # Add new tokens to added_tokens array
    for token_str, token_id in tokenizer.special_tokens.items():
        if token_id not in existing_token_ids:
            # Determine if this is a special token
            is_special = token_str.startswith("<") and token_str.endswith(">")
            
            token_entry = {
                "id": token_id,
                "content": token_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": is_special
            }
            
            tokenizer_data["added_tokens"].append(token_entry)
    
    # Sort added_tokens by ID for consistency
    tokenizer_data["added_tokens"].sort(key=lambda x: x["id"])
    
    # Save updated tokenizer.json
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    log.info(f"Updated tokenizer.json added_tokens with {len(tokenizer_data['added_tokens'])} total tokens")