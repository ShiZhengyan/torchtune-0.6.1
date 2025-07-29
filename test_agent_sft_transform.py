#!/usr/bin/env python3

import torch
from torchtune.models.qwen2_5 import qwen2_5_tokenizer
from torchtune.datasets._agent_sft import AgentSFTTransform
from torchtune.data._messages import OpenAIToMessages

def test_agent_sft_transform():
    """Test AgentSFTTransform parsing of reasoning_mask and tool_call_mask."""
    
    # Initialize Qwen 2.5 tokenizer
    print("Initializing Qwen 2.5 tokenizer...")
    tokenizer = qwen2_5_tokenizer(
        path="/home/zhengyanshi/project/torchtune-0.6.1/llm-weights/Qwen/Qwen2.5-0.5B-Instruct/vocab.json",
        merges_file="/home/zhengyanshi/project/torchtune-0.6.1/llm-weights/Qwen/Qwen2.5-0.5B-Instruct/merges.txt"
    )
    
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
            # {
            # "content": "OBSERVATION:\nWARNING: replace_in_file didn't find pattern 'Pattern' in 'test.txt' file.\r\nPattern not found result: False\r\nPattern found result: None",
            # "role": "user"
            # },
            # {
            # "content": "The reproduction confirms the issue - when the pattern is not found, it returns `False`, but when the pattern is found and replaced, it returns `None`. This is inconsistent behavior.\n\nLet's fix this by making it return `True` when the pattern is found and replaced:\n\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/conan/tools/files/files.py</parameter>\n<parameter=old_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n</parameter>\n<parameter=new_str>\n    content = content.replace(search, replace)\n    save(conanfile, file_path, content, encoding=encoding)\n    return True\n</parameter>\n</function>",
            # "role": "assistant"
            # },
            {
            "role": "user",
            "content": "<tool_response>\nThe file `pandas/io/stata.py` has been updated successfully.\n\nDiff:\n\n--- original\n+++ current\n@@ -303,7 +303,7 @@\n \n     elif fmt.startswith((\"%tq\", \"tq\")):\n         # Delta quarters relative to base\n-        ordinals = dates + stata_epoch.year * 4\n+        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 4\n         res = np.array(ordinals, dtype=\"M8[3M]\").astype(\"M8[s]\")\n         return Series(res, index=dates.index)\n \n\n</tool_response>"
            },
            {
            "role": "assistant",
            "content": "Now let me run the tests to verify that the fix works:\n\n<tool_call>\n{\"name\": \"eval\", \"arguments\": {}}\n</tool_call>"
            }
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
        # train_on_tool_calls_only=True,
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
        print("DETAILED TOKEN ANALYSIS")
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
    test_agent_sft_transform()