import gradio as gr
import subprocess
import os

def trigger_pipeline():
    # In HF spaces, the working directory is the root of the repository
    try:
        # Run rank.py and capture the exact profiler output
        result = subprocess.run(["python", "rank.py"], capture_output=True, text=True, check=True)
        
        # Check if submission.csv was created
        csv_path = "submission.csv" if os.path.exists("submission.csv") else None
        
        return "Pipeline Execution Successful!\n\n" + result.stdout, csv_path
    except subprocess.CalledProcessError as e:
        return f"Pipeline Failed!\n\nError:\n{e.stderr}\n\nOutput:\n{e.stdout}", None

if __name__ == "__main__":
    iface = gr.Interface(
        fn=trigger_pipeline,
        inputs=[],
        outputs=[
            gr.Textbox(lines=30, label="Sandbox Profiler Logs"),
            gr.File(label="Download Final submission.csv")
        ],
        title="India Runs AI Challenge - Trail 8 Offline Ranker",
        description="Click 'Submit' to run the offline 4B Generative LLM pipeline (No Internet, CPU Only, 16GB RAM limit)."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)
