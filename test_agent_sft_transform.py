#!/usr/bin/env python3

import torch
import argparse
import os
from torchtune.models.qwen2_5 import qwen2_5_tokenizer
from torchtune.datasets._agent_sft import AgentSFTTransform
from torchtune.data._messages import OpenAIToMessages
from transformers import AutoTokenizer


def get_tokenizer(tokenizer_type, tokenizer_path):
    """Get tokenizer based on type (torchtune or huggingface)."""
    if tokenizer_type == "hf":
        print(f"Loading HuggingFace tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer
    
    elif tokenizer_type == "tt":
        print(f"Loading TorchTune tokenizer from: {tokenizer_path}")
        vocab_path = os.path.join(tokenizer_path, "vocab.json")
        merges_path = os.path.join(tokenizer_path, "merges.txt")
        special_tokens_path = os.path.join(tokenizer_path, "tokenizer.json")
        
        tokenizer = qwen2_5_tokenizer(
            path=vocab_path,
            merges_file=merges_path,
            special_tokens_path=special_tokens_path,
        )
        return tokenizer
    
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Choose 'tt' or 'hf'.")

def test_hf_tokenization(tokenizer_path):
    """Test regular HuggingFace tokenization approach without transforms."""
    
    # Initialize HF tokenizer
    print(f"Initializing HuggingFace tokenizer...")
    tokenizer = get_tokenizer("hf", tokenizer_path)
    
    # Sample data following your example
    sample_data = {
        "messages": [
            {
                "content": "OBSERVATION:\nFile created successfully at: /testbed/reproduce.py",
                "role": "user"
            },
            {
                "content": "Let's run the reproduction script:\n\n<function=bash>\n<parameter=command>cd /testbed && python reproduce.py</parameter>\n</function>The is the reasoning end",
                "role": "assistant"
            },
            {
            "content": "OBSERVATION:\nWARNING: replace_in_file didn't find pattern 'Pattern' in 'test.txt' file.\r\nPattern not found result: False\r\nPattern found result: None",
            "role": "user"
            },
            {
            "content": "The reproduction confirms the issue - when the pattern is not found, it returns `False`, but when the pattern is found and replaced, it returns `None`. This is inconsistent behavior.\n\nLet's fix this by making it return `True` when the pattern is found and replaced:\n\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/conan/tools/files/files.py</parameter>\n<parameter=old_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n</parameter>\n<parameter=new_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n    return True\n</parameter>\n</function>",
            "role": "assistant"
            },
        ]
    }
    
    print("\nOriginal sample data:")
    for i, msg in enumerate(sample_data["messages"]):
        print(f"Message {i} ({msg['role']}): {repr(msg['content'])}")
    
    # Apply regular HF tokenization
    print("\nApplying regular HuggingFace tokenization...")
    try:
        # Convert messages to text format for tokenization
        formatted_text = ""
        for msg in sample_data["messages"]:
            role_prefix = f"<|im_start|>{msg['role']}\n" if hasattr(tokenizer, 'chat_template') else f"{msg['role']}: "
            role_suffix = "<|im_end|>\n" if hasattr(tokenizer, 'chat_template') else "\n\n"
            formatted_text += role_prefix + msg['content'] + role_suffix
        
        print(f"\nFormatted text for tokenization:")
        print(repr(formatted_text))
        
        # Tokenize the text
        tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
        
        print(f"\nTotal tokens: {len(tokens)}")
        
        # Print detailed analysis
        print("\n" + "="*80)
        print("DETAILED TOKEN ANALYSIS (HuggingFace)")
        print("="*80)
        print(f"{'Index':<6} {'Token ID':<10} {'Token Text':<30}")
        print("-" * 50)
        
        for i, token_id in enumerate(tokens):
            # Decode individual token
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                # Clean up the token text for display
                token_text = repr(token_text)[:25] + ("..." if len(repr(token_text)) > 25 else "")
            except:
                token_text = "<decode_error>"
            
            print(f"{i:<6} {token_id:<10} {token_text:<30}")
        
        # Show the reconstructed text
        print("\n" + "="*80)
        print("RECONSTRUCTED TEXT FROM TOKENS")
        print("="*80)
        reconstructed = tokenizer.decode(tokens, skip_special_tokens=False)
        print(repr(reconstructed))
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        import traceback
        traceback.print_exc()


def test_agent_sft_transform(tokenizer_path):
    """Test AgentSFTTransform parsing of reasoning_mask and tool_call_mask with TorchTune tokenizer."""
    
    # Initialize TT tokenizer
    print(f"Initializing TorchTune tokenizer...")
    tokenizer = get_tokenizer("tt", tokenizer_path)
    
    # Sample data following your example
    sample_data = {
        "messages": [
            {
                "content": "OBSERVATION:\nFile created successfully at: /testbed/reproduce.py",
                "role": "user"
            },
            {
                "content": "Let's run the reproduction script:\n\n<function=bash>\n<parameter=command>cd /testbed && python reproduce.py</parameter>\n</function>The is the reasoning end",
                "role": "assistant"
            },
            {
            "content": "OBSERVATION:\nWARNING: replace_in_file didn't find pattern 'Pattern' in 'test.txt' file.\r\nPattern not found result: False\r\nPattern found result: None",
            "role": "user"
            },
            {
            "content": "The reproduction confirms the issue - when the pattern is not found, it returns `False`, but when the pattern is found and replaced, it returns `None`. This is inconsistent behavior.\n\nLet's fix this by making it return `True` when the pattern is found and replaced:\n\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/conan/tools/files/files.py</parameter>\n<parameter=old_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n</parameter>\n<parameter=new_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n    return True\n</parameter>\n</function>",
            "role": "assistant"
            },
            # {
            # "role": "user",
            # "content": "<tool_response>\nThe file `pandas/io/stata.py` has been updated successfully.\n\nDiff:\n\n--- original\n+++ current\n@@ -303,7 +303,7 @@\n \n     elif fmt.startswith((\"%tq\", \"tq\")):\n         # Delta quarters relative to base\n-        ordinals = dates + stata_epoch.year * 4\n+        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 4\n         res = np.array(ordinals, dtype=\"M8[3M]\").astype(\"M8[s]\")\n         return Series(res, index=dates.index)\n \n\n</tool_response>"
            # },
            # {
            # "role": "assistant",
            # "content": "Now let me run the tests to verify that the fix works:\n\n<tool_call>\n{\"name\": \"eval\", \"arguments\": {}}\n</tool_call>"
            # }
        ]
    }
    
    print("\nOriginal sample data:")
    for i, msg in enumerate(sample_data["messages"]):
        print(f"Message {i} ({msg['role']}): {repr(msg['content'])}")
    
    # Create transforms
    message_transform = OpenAIToMessages(
        train_on_input=False,
        column_map={"messages": "messages"},
    )
    
    agent_transform = AgentSFTTransform(
        message_transform=message_transform,
        model_transform=tokenizer,
        enable_token_classification=True,
        # train_on_tool_calls_only=False,
        train_on_reasoning_only=True,
        # tool_call_start_token="<tool_call>",
        # tool_call_end_token="</tool_call>",
    )
    
    # Apply transform
    print("\nApplying AgentSFTTransform...")
    try:
        result = agent_transform(sample_data)
        
        # Extract information
        tokens = result["tokens"]
        labels = result.get("labels", [])
        reasoning_mask = result.get("reasoning_mask", [])
        tool_call_mask = result.get("tool_call_mask", [])
        
        print(f"\nTotal tokens: {len(tokens)}")
        print(f"Labels length: {len(labels)}")
        print(f"Reasoning mask length: {len(reasoning_mask)}")
        print(f"Tool call mask length: {len(tool_call_mask)}")
        
        # Print detailed analysis
        print("\n" + "="*80)
        print("DETAILED TOKEN ANALYSIS (TorchTune)")
        print("="*80)
        print(f"{'Index':<6} {'Token ID':<10} {'Token Text':<25} {'Label':<10} {'Reasoning':<10} {'Tool Call':<10}")
        print("-" * 80)
        
        for i in range(len(tokens)):
            token_id = tokens[i]
            
            # Decode individual token
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                # Clean up the token text for display
                token_text = repr(token_text)[:20] + ("..." if len(repr(token_text)) > 20 else "")
            except:
                token_text = "<decode_error>"
            
            label = labels[i] if i < len(labels) else "N/A"
            reasoning = reasoning_mask[i] if i < len(reasoning_mask) else "N/A"
            tool_call = tool_call_mask[i] if i < len(tool_call_mask) else "N/A"
            
            print(f"{i:<6} {token_id:<10} {token_text:<25} {label:<10} {reasoning:<10} {tool_call:<10}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        if reasoning_mask:
            reasoning_count = sum(reasoning_mask)
            tool_call_count = sum(tool_call_mask)
            neither_count = len(reasoning_mask) - reasoning_count - tool_call_count
            
            print(f"Total tokens: {len(tokens)}")
            print(f"Reasoning tokens: {reasoning_count} ({reasoning_count/len(tokens)*100:.1f}%)")
            print(f"Tool call tokens: {tool_call_count} ({tool_call_count/len(tokens)*100:.1f}%)")
            print(f"Neither (special tokens): {neither_count} ({neither_count/len(tokens)*100:.1f}%)")
        
        # Check for function blocks in the original text
        full_text = "\n".join([msg["content"] for msg in sample_data["messages"]])
        print(f"\nOriginal full text contains '<function'?: {'<function' in full_text}")
        print(f"Original full text contains '</function>'?: {'</function>' in full_text}")
        
        # Show the reconstructed text
        print("\n" + "="*80)
        print("RECONSTRUCTED TEXT FROM TOKENS")
        print("="*80)
        reconstructed = tokenizer.decode(tokens, skip_special_tokens=False)
        print(repr(reconstructed))
        
    except Exception as e:
        print(f"Error during transform: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenization with Qwen 2.5 tokenizer")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="llm-weights/Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to the tokenizer directory containing vocab.json and merges.txt (default: llm-weights/Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="tt",
        choices=["tt", "hf"],
        help="Type of tokenizer to use: 'tt' for TorchTune or 'hf' for HuggingFace (default: tt)"
    )
    args = parser.parse_args()
    
    if args.tokenizer_type == "hf":
        test_hf_tokenization(args.tokenizer_path)
    else:
        test_agent_sft_transform(args.tokenizer_path)