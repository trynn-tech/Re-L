# engine/orchestrator.py 
import re, os
from engine.agent import hegelian_qa
from engine.tools import write_file, exec_python

def run_objective(goal: str, max_iterations: int = 10):
    current_task = goal
    
    for i in range(max_iterations):
        print(f"\n--- 🔄 Iteration {i+1} ---")
        response = hegelian_qa(current_task)
        
        # --- 1. ACTION PARSING ---
        # Look for the official tag first, then the fallback markdown
        path = None
        content = None
        
        action_match = re.search(r'\[ACTION: WRITE_FILE path="([^"]+)"\](.*?)\[/ACTION\]', response, re.DOTALL)
        if action_match:
            path = action_match.group(1)
            content = action_match.group(2).strip()
        else:
            code_block = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
            if code_block:
                path = "workspace/logic/entropy.py"
                content = code_block.group(1).strip()

        # --- 2. EXECUTION & VALIDATION ---
        if path and content:
            write_file(path, content)
            # Run the file and get stdout/stderr
            execution_result = exec_python(path)
            print(f"🧪 Execution Output: {execution_result}")

            # VALIDATION LOGIC: Did it actually meet the math requirements?
            # We look for the specific expected values in the output
            is_valid = "0" in str(execution_result) and ("1.58" in str(execution_result) or "1.59" in str(execution_result))
            
            if is_valid:
                print("✅ [SYSTEM] Math Verified. Task complete.")
                # Feed a completion signal back to the agent
                current_task = f"SUCCESS: The code in {path} is verified. Terminate with [STATUS: COMPLETE]."
            else:
                print("❌ [SYSTEM] Math Mismatch. Feedback sent to agent.")
                current_task = (
                    f"EXECUTION FAILURE: The code at {path} did not return the expected values.\n"
                    f"Target: H('aaaaa')=0, H('abcd')~=1.58\n"
                    f"Actual Output: {execution_result}\n"
                    "Please rewrite the logic to fix the Shannon Entropy formula."
                )
            continue

        # --- 3. TERMINATION ---
        if "[STATUS: COMPLETE]" in response:
            print("🏁 Objective met. Halting.")
            break
            
        current_task = f"Previous Thought: {response}. We are NOT finished. Proceed to goal: {goal}"
