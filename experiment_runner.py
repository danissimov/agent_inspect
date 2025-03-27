import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from inspect_ai import eval
from typing import List, Dict, Any, Optional, Union

# Import your task definition
from directories_task import nested_dirs_challenge

class ExperimentRunner:
    """Class to automate running multiple LLM evaluation experiments with AISI Inspect."""
    
    def __init__(self, 
                output_dir: str = "experiment_results",
                models: List[str] = None, 
                depths: List[int] = None,
                experiment_name: str = None):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
            models: List of model identifiers to evaluate
            depths: List of depth values to test
            experiment_name: Optional name for this experiment run
        """
        self.output_dir = output_dir
        self.models = models or ["openai/gpt-4o-mini", "openai/gpt-4o", 
                              "google/gemini-1.5-flash", "google/gemini-1.5-pro"]
        self.depths = depths or [2, 3, 4]
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def run_experiments(self, 
                      prompts: Optional[Dict[str, str]] = None,
                      run_description: str = "Standard experiment run"):
        """
        Run a set of experiments with different models, depths, and prompts.
        
        Args:
            prompts: Dictionary of prompt_name -> prompt_text for different prompts to test
            run_description: Description of this experiment run
        """
        prompts = prompts or {"default": None}  # Default prompt in task
        
        # Track command counts across runs
        bash_command_count = 0
        
        # Iterate through all combinations
        for model in self.models:
            for depth in self.depths:
                for prompt_name, prompt in prompts.items():
                    print(f"Running experiment: model={model}, depth={depth}, prompt={prompt_name}")
                    
                    # Reset command count for each run
                    bash_command_count = 0
                    
                    # Run evaluation with current settings
                    try:
                        # If a prompt is provided, we'd need to modify the task to use it
                        # For now, we're just using the default task with different depths
                        log = eval(nested_dirs_challenge(depth_n=depth),
                                model=model,    
                                metadata={
                                    "run_info": f"""
                                            Run: {self.experiment_name}
                                            Model: {model}
                                            Depth: {depth}
                                            Prompt: {prompt_name}
                                            Description: {run_description}
                                            """,
                                    "bash_command_count": bash_command_count
                                })
                        
                        # Process results and store them
                        self._process_log(log, model, depth, prompt_name)
                        
                    except Exception as e:
                        print(f"Error running experiment: {e}")
                        # Record the failure
                        self.results.append({
                            "model": model,
                            "depth": depth,
                            "prompt": prompt_name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Save all results to disk
        self._save_results()
        
    def _process_log(self, log, model: str, depth: int, prompt_name: str):
        """
        Process a log from an experiment run and extract metrics.
        
        Args:
            log: The log object returned from eval()
            model: Model identifier
            depth: Depth value used
            prompt_name: Name of the prompt used
        """
        # Extract scores and metrics from the log
        try:
            samples = log.samples
            
            for i, sample in enumerate(samples):
                # Extract the core metrics for each sample
                result_data = {
                    "model": model,
                    "depth": depth,
                    "prompt": prompt_name,
                    "sample_id": i,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
                
                # Add aggregated scores (overall performance)
                result_data["overall_score"] = sample.score
                
                # Extract detailed metrics from the scorers
                if hasattr(sample, "scores") and sample.scores:
                    # Extract check_nested_dirs metrics
                    if "check_nested_dirs" in sample.scores:
                        nested_dirs_score = sample.scores["check_nested_dirs"]
                        result_data["nested_dirs_value"] = nested_dirs_score.get("value", 0)
                        
                        # Extract individual component scores if available
                        if "metrics" in nested_dirs_score:
                            metrics = nested_dirs_score["metrics"]
                            for key, value in metrics.items():
                                result_data[f"nested_dirs_{key}"] = value
                        
                        # Try to parse the explanation string if metrics aren't available
                        elif "explanation" in nested_dirs_score:
                            explanation = nested_dirs_score["explanation"]
                            result_data["depth_passed"] = "✓" in explanation.split(",")[0]
                            result_data["breadth_passed"] = "✓" in explanation.split(",")[1]
                            result_data["uniqueness_passed"] = "✓" in explanation.split(",")[2]
                    
                    # Extract command_efficiency_scorer metrics
                    if "command_efficiency_scorer" in sample.scores:
                        efficiency_score = sample.scores["command_efficiency_scorer"]
                        result_data["efficiency_value"] = efficiency_score.get("value", 0)
                        
                        # Extract command counts if available
                        if "metrics" in efficiency_score:
                            metrics = efficiency_score["metrics"]
                            for key, value in metrics.items():
                                # Skip storing the full commands list in the results table
                                if key != "commands_list":
                                    result_data[f"efficiency_{key}"] = value
                
                # Add to results
                self.results.append(result_data)
                
        except Exception as e:
            print(f"Error processing log: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_results(self):
        """Save experiment results to disk in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(self.output_dir, f"{self.experiment_name}")
        
        # Save as JSON
        json_path = f"{base_path}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV using pandas
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(f"{base_path}.csv", index=False)
            
            # Also save an Excel file for easier viewing
            df.to_excel(f"{base_path}.xlsx", index=False)
            
            print(f"Saved results to {base_path}.json, {base_path}.csv, and {base_path}.xlsx")
        except Exception as e:
            print(f"Error saving to DataFrame: {e}")
    
    def generate_visualizations(self):
        """Generate visualizations of the experiment results."""
        if not self.results:
            print("No results to visualize")
            return
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Create output directory for visualizations
            viz_dir = os.path.join(self.output_dir, f"{self.experiment_name}_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Overall performance by model
            plt.figure(figsize=(12, 6))
            df_model = df.groupby('model')['overall_score'].mean().reset_index()
            plt.bar(df_model['model'], df_model['overall_score'])
            plt.title('Overall Score by Model')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'model_performance.png'))
            plt.close()
            
            # 2. Performance by depth
            plt.figure(figsize=(10, 6))
            df_depth = df.groupby(['model', 'depth'])['overall_score'].mean().reset_index()
            for model in df_depth['model'].unique():
                model_data = df_depth[df_depth['model'] == model]
                plt.plot(model_data['depth'], model_data['overall_score'], marker='o', label=model)
            plt.title('Score by Directory Depth')
            plt.xlabel('Depth')
            plt.ylabel('Average Score')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(viz_dir, 'depth_performance.png'))
            plt.close()
            
            # 3. Component scores by model (depth, breadth, uniqueness)
            if 'nested_dirs_depth_score' in df.columns:
                plt.figure(figsize=(14, 8))
                components = ['nested_dirs_depth_score', 'nested_dirs_breadth_score', 'nested_dirs_uniqueness_score']
                df_components = df.groupby('model')[components].mean().reset_index()
                
                x = range(len(df_components['model']))
                width = 0.25
                
                plt.bar([i - width for i in x], df_components['nested_dirs_depth_score'], 
                      width=width, label='Depth')
                plt.bar(x, df_components['nested_dirs_breadth_score'], 
                      width=width, label='Breadth')
                plt.bar([i + width for i in x], df_components['nested_dirs_uniqueness_score'], 
                      width=width, label='Uniqueness')
                
                plt.title('Component Scores by Model')
                plt.xlabel('Model')
                plt.ylabel('Average Score')
                plt.xticks(x, df_components['model'], rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'component_scores.png'))
                plt.close()
            
            # 4. Command efficiency by model and depth
            if 'efficiency_value' in df.columns:
                plt.figure(figsize=(12, 6))
                df_efficiency = df.groupby(['model', 'depth'])['efficiency_value'].mean().reset_index()
                
                for model in df_efficiency['model'].unique():
                    model_data = df_efficiency[df_efficiency['model'] == model]
                    plt.plot(model_data['depth'], model_data['efficiency_value'], marker='o', label=model)
                
                plt.title('Command Efficiency by Depth')
                plt.xlabel('Depth')
                plt.ylabel('Efficiency Score')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(viz_dir, 'efficiency_score.png'))
                plt.close()
            
            print(f"Visualizations saved to {viz_dir}")
            
            # Generate summary statistics
            summary = df.groupby(['model', 'depth']).agg({
                'overall_score': ['mean', 'std', 'min', 'max'],
                'nested_dirs_value': ['mean', 'std'] if 'nested_dirs_value' in df.columns else ['count'],
                'efficiency_value': ['mean', 'std'] if 'efficiency_value' in df.columns else ['count']
            }).reset_index()
            
            summary.to_csv(os.path.join(viz_dir, 'summary_statistics.csv'))
            summary.to_excel(os.path.join(viz_dir, 'summary_statistics.xlsx'))
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    
    prompt_variations = {
        "default": None,  # Use default prompt
        "concise": "Create a nested directory structure with exactly n subdirectories at each level, down to depth n. Each directory must have a unique name.",
        "detailed": "Create a nested directory structure. The structure should have exactly n subdirectories at each level, and the depth should be exactly n levels deep. All directory names must be unique. Use a single mkdir command if possible."
    }
    
    runner = ExperimentRunner(
        output_dir="experiment_results",
        models=["openai/gpt-4o-mini", "openai/gpt-4o"],  # Limit models for quicker test run
        depths=[2, 3],  # Test with depths 2 and 3
        experiment_name="directory_structure_test"
    )
    
    runner.run_experiments(
        prompts=prompt_variations,
        run_description="Testing different prompt variations for directory structure task"
    )
    
    runner.generate_visualizations() 