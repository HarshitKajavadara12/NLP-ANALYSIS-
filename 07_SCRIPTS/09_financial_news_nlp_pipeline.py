"""
Financial News NLP Pipeline - Main Orchestrator
===============================================

End-to-end pipeline executing all 8 analysis steps:
1. Data Collection & Preprocessing
2. Exploratory Data Analysis
3. Sentiment Analysis
4. Named Entity Recognition
5. Event Detection & Classification
6. Topic Modeling & Classification
7. Advanced Feature Engineering
8. Ensemble Modeling Strategies

Run this script to execute the complete analysis pipeline.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time
import json

# Setup paths
BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / '07_SCRIPTS'
RESULTS_DIR = BASE_DIR / '03_RESULTS'
LOGS_DIR = BASE_DIR / '05_LOGS'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline scripts in execution order
PIPELINE_SCRIPTS = [
    '01_data_collection_and_preprocessing.py',
    '02_exploratory_data_analysis.py',
    '03_sentiment_analysis_comprehensive.py',
    '04_named_entity_recognition_comprehensive.py',
    '05_event_detection_classification_comprehensive.py',
    '06_topic_modeling_classification_comprehensive.py',
    '07_advanced_feature_engineering.py',
    '08_ensemble_modeling_strategies.py'
]

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*100)
    print(f"  {text}")
    print("="*100 + "\n")

def print_step(step_num, total, script_name):
    """Print step information"""
    print(f"\n{'='*100}")
    print(f"  STEP {step_num}/{total}: {script_name}")
    print(f"{'='*100}\n")

def run_script(script_path):
    """Run a Python script and capture output"""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def save_execution_log(log_data):
    """Save execution log to JSON"""
    log_file = LOGS_DIR / f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    return log_file

def main():
    """Main pipeline execution"""
    
    print_header("FINANCIAL NEWS NLP ANALYSIS PIPELINE")
    print(f" Base Directory: {BASE_DIR}")
    print(f" Scripts Directory: {SCRIPTS_DIR}")
    print(f" Results Directory: {RESULTS_DIR}")
    print(f" Logs Directory: {LOGS_DIR}")
    print(f"\n  Pipeline Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n Pipeline Steps: {len(PIPELINE_SCRIPTS)}")
    for idx, script in enumerate(PIPELINE_SCRIPTS, 1):
        print(f"   {idx}. {script}")
    
    # Execution tracking
    execution_log = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_steps': [],
        'total_steps': len(PIPELINE_SCRIPTS),
        'successful_steps': 0,
        'failed_steps': 0,
        'warnings': []
    }
    
    overall_start = time.time()
    
    # Execute each script in sequence
    for idx, script_name in enumerate(PIPELINE_SCRIPTS, 1):
        print_step(idx, len(PIPELINE_SCRIPTS), script_name)
        
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            error_msg = f" Script not found: {script_path}"
            print(error_msg)
            execution_log['warnings'].append(error_msg)
            execution_log['failed_steps'] += 1
            continue
        
        step_start = time.time()
        print(f"  Executing: {script_name}...")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Run script
        success, stdout, stderr = run_script(script_path)
        
        step_duration = time.time() - step_start
        
        # Log step execution
        step_log = {
            'step_number': idx,
            'script_name': script_name,
            'start_time': datetime.fromtimestamp(step_start).strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round(step_duration, 2),
            'duration_formatted': f"{int(step_duration // 60)}m {int(step_duration % 60)}s",
            'success': success
        }
        
        if success:
            print(f"\n SUCCESS: {script_name}")
            print(f"   Duration: {step_log['duration_formatted']}")
            execution_log['successful_steps'] += 1
            
            # Show last few lines of output
            output_lines = stdout.strip().split('\n')
            if len(output_lines) > 10:
                print(f"\n Last 10 lines of output:")
                for line in output_lines[-10:]:
                    print(f"   {line}")
        else:
            print(f"\n FAILED: {script_name}")
            print(f"   Duration: {step_log['duration_formatted']}")
            print(f"\n Error Output:")
            print(stderr)
            execution_log['failed_steps'] += 1
            step_log['error'] = stderr
            
            # Ask if user wants to continue
            if idx < len(PIPELINE_SCRIPTS):
                response = input(f"\n  Continue to next step? (y/n): ")
                if response.lower() != 'y':
                    print("\n Pipeline execution stopped by user.")
                    break
        
        execution_log['pipeline_steps'].append(step_log)
        
        # Separator between steps
        if idx < len(PIPELINE_SCRIPTS):
            print("\n" + "-"*100 + "\n")
            time.sleep(1)  # Brief pause between steps
    
    # Calculate overall statistics
    overall_duration = time.time() - overall_start
    execution_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    execution_log['total_duration_seconds'] = round(overall_duration, 2)
    execution_log['total_duration_formatted'] = f"{int(overall_duration // 3600)}h {int((overall_duration % 3600) // 60)}m {int(overall_duration % 60)}s"
    execution_log['success_rate'] = round((execution_log['successful_steps'] / execution_log['total_steps']) * 100, 2)
    
    # Save execution log
    log_file = save_execution_log(execution_log)
    
    # Print final summary
    print_header("PIPELINE EXECUTION COMPLETE")
    
    print(" EXECUTION SUMMARY:")
    print("="*100)
    print(f"  Total Steps: {execution_log['total_steps']}")
    print(f"  Successful: {execution_log['successful_steps']} ")
    print(f"  Failed: {execution_log['failed_steps']} ")
    print(f"  Success Rate: {execution_log['success_rate']}%")
    print(f"  Total Duration: {execution_log['total_duration_formatted']}")
    print("="*100)
    
    print(f"\n  TIMING BREAKDOWN:")
    print("="*100)
    for step in execution_log['pipeline_steps']:
        status_icon = "" if step['success'] else ""
        print(f"  {step['step_number']}. {step['script_name']:50s} {status_icon}  {step['duration_formatted']}")
    print("="*100)
    
    # Step-by-step results
    print(f"\n STEP DETAILS:")
    print("="*100)
    for step in execution_log['pipeline_steps']:
        status = "SUCCESS" if step['success'] else "FAILED"
        print(f"\n  Step {step['step_number']}: {step['script_name']}")
        print(f"  Status: {status}")
        print(f"  Duration: {step['duration_formatted']}")
        if not step['success'] and 'error' in step:
            print(f"  Error: {step['error'][:200]}...")
    print("="*100)
    
    # Recommendations
    print(f"\n NEXT STEPS:")
    print("="*100)
    if execution_log['successful_steps'] == execution_log['total_steps']:
        print("   All pipeline steps completed successfully!")
        print(f"\n   Check results in:")
        print(f"     - {BASE_DIR / '01_DATA'} (processed datasets)")
        print(f"     - {BASE_DIR / '03_RESULTS'} (analysis results)")
        print(f"     - {BASE_DIR / '04_MODELS'} (trained models)")
        print(f"     - {BASE_DIR / '01_DATA' / 'visualizations'} (charts)")
        print(f"\n   Key outputs:")
        print(f"     - Final feature matrix: 01_DATA/features/final_feature_matrix.csv")
        print(f"     - Trained models: 04_MODELS/ensemble/stacking.pkl")
        print(f"     - Model performance: 03_RESULTS/all_model_performance.csv")
        print(f"\n   Ready for production deployment!")
        print(f"     - Use trained models for real-time sentiment prediction")
        print(f"     - Generate trading signals from new articles")
        print(f"     - Monitor model performance with new data")
    else:
        print(f"    {execution_log['failed_steps']} step(s) failed.")
        print(f"  Review the error messages above and:")
        print(f"     1. Check if all required packages are installed")
        print(f"     2. Verify data files are accessible")
        print(f"     3. Ensure sufficient disk space and memory")
        print(f"     4. Re-run failed scripts individually for detailed debugging")
        print(f"\n   Execution log saved to: {log_file}")
    print("="*100)
    
    print(f"\n Full execution log saved to: {log_file}")
    print(f"\n  Pipeline End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*100}\n")
    
    return execution_log['successful_steps'] == execution_log['total_steps']

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n Pipeline execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n Unexpected error during pipeline execution:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
