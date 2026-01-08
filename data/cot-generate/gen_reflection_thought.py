import json
import os
import traceback
import datetime 
import backoff
from dotenv import load_dotenv
from module.reflector import gen_reflection_thought_simple

load_dotenv()

def generate_all_history(previous_steps):
    previous_actions = [step['value']['action'] for step in previous_steps]

    if not previous_actions:
        return "None"
    
    history = ""
    for i in range(len(previous_actions)):
        history += f"Step {i+1}: {previous_actions[i]}\n"

    return history

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_time=180,  # Increase max_time to allow more retries
    max_tries=2,  # Limit the number of retries to prevent infinite loops
    jitter=backoff.full_jitter,  # Add jitter to spread out retry attempts
)
def generate_reflection_thought_batch(
    vlm_output_dir: str,
    model:str,
    timestamp: str,
    server_addr: str,
    port:str,
    ) -> dict:

    try:
        # 1. task 전체 내용을 읽어 온다
        # 2. 각 step 별로 reflection thought 를 생성한다.
        json_file_list = os.listdir(vlm_output_dir)    
        json_data = dict()
        for filename in json_file_list:
            json_file_path = os.path.join(vlm_output_dir, filename)
            with open(json_file_path, 'r') as f: 
                json_data[filename] = json.load(f)        

        # @학선님
        # 00.json, 01.json, 02.json, ... , meta.json (아마도 파일명이 맞을거예요)
        # 이런 형태로 저장 되어있습니다. 

        meta_data = json_data['meta.json']
        instruction = meta_data['instruction']

        json_file_list.sort()
        target_file = 'meta.json'
        if target_file in json_file_list:
            index_to_pop = json_file_list.index(target_file)
            json_file_list.pop(index_to_pop)

        previous_steps = list()
        for this_step, next_step in zip(json_file_list, json_file_list[1:]):
            this_step_data = json_data[this_step]
            next_step_data = json_data[next_step]
            history_steps = generate_all_history(previous_steps)
            

            previous_steps.append(this_step_data)

            reflect_response = gen_reflection_thought_simple(
                model=model,
                goal=instruction,
                history_steps=history_steps,
                current_step=this_step_data['value'],
                next_step=next_step_data['value'],
                server_addr=server_addr,
                port=port,
            )

            this_step_data['value']['this_step_unnecessary'] = reflect_response['this_step_unnecessary']
            this_step_data['value']['reflection'] = reflect_response['reflection']

            with open(os.path.join(vlm_output_dir, this_step), 'w') as f: 
                json.dump(this_step_data, f, indent=4)
    
    except Exception as e:
        print("=" * 100)
        print(f"Unexpected Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("=" * 100)
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Inner Monologue")
    parser.add_argument("--vlm_output_dir", type=str, default="./gen_cot_example/output/tasks", help="Directory for generated files from VLM")
    parser.add_argument("--model", type=str, default="SmolLM3-3B", help="Model to use for LLM calls")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--server_addr", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=str, default='7100')
    
    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())    
    generate_reflection_thought_batch(**kwargs)

if __name__ == "__main__":
    main()
