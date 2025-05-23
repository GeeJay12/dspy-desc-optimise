import json
import logging
import os
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

import dspy

logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # Or logging.CRITICAL
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)  # Or logging.CRITICAL
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)  # Or logging.CRITICAL

# Configure your application's logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Pydantic Schema Definition ---
# This mirrors the relevant part of your schema for this example.


class Illness(str, Enum):
    DIABETES = "Diabetes"
    LIVER_CIRRHOSIS = "Liver Cirrhosis"
    ANXIETY_DISORDER = "Anxiety Disorder"
    ASTHMA = "Asthma"
    PSORIASIS = "Psoriasis"
    CHRONIC_KIDNEY_DISEASE = "Chronic Kidney Disease"
    PNEUMONIA = "Pneumonia"
    ANEMIA = "Anemia"
    HYPOTHYROIDISM = "Hypothyroidism"
    HYPERTENSION = "Hypertension"
    HEART_DISEASE = "Heart Disease"
    CANCER = "Cancer"
    BRONCHITIS = "Bronchitis"
    SINUSITIS = "Sinusitis"
    PEPTIC_ULCER = "Peptic Ulcer"
    EPLEPSY = "Epilepsy"
    OTITIS_MEDIA = "Otitis Media"
    DERMATITIS = "Dermatitis"
    URINARY_TRACT_INFECTION = "Urinary Tract Infection"
    TUBERCULOSIS = "Tuberculosis"
    DEPRESSION = "Depression"
    HYPERTHYROIDISM = "Hyperthyroidism"
    GASTROENTERITIS = "Gastroenteritis"


class IllnessExtraction(BaseModel):
    """Schema for extracting illness information."""

    illness: Illness = Field(
        description="Derive the appropriate illness from the context",  # Initial description to be optimized
    )


# --- DSPy Signature Definition ---
class SymptomToIllnessSignature(dspy.Signature):
    """
    Given a patient's description of their symptoms, identify the most probable medically relevant illness.
    The instruction for extracting the illness will be optimized.
    """

    symptom_context: str = dspy.InputField(
        desc="Patient's description of their symptoms."
    )
    extracted_illness: str = dspy.OutputField(
        desc="The name of the medically relevant illness."
    )

    # The initial instruction for the 'extracted_illness' field will be dynamically set
    # from the Pydantic schema's description during module initialization.


# --- DSPy Module ---
class IllnessExtractorModule(dspy.Module):
    def __init__(self, initial_instruction: str):
        super().__init__()
        # We create a Predict module using a copy of the signature
        # and then update its instruction.
        self.predictor = dspy.Predict(SymptomToIllnessSignature)

        # Get the current signature instance from the predictor
        current_signature_instance = self.predictor.signature

        # Create a new signature instance with the updated instruction
        # using the .with_instructions() method.
        self.predictor.signature = current_signature_instance.with_instructions(
            initial_instruction
        )

        logger.info(
            f"Initial instruction for IllnessExtractorModule set to: {self.predictor.signature.instructions}"
        )

    def forward(self, symptom_context: str) -> dspy.Prediction:
        return self.predictor(symptom_context=symptom_context)


# --- Data Loading Function ---
def load_training_data(json_path: str) -> List[dspy.Example]:
    """Loads training data from a JSON file."""
    examples = []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if "data" not in data or not isinstance(data["data"], list):
            logger.error("JSON data must contain a 'data' key with a list of examples.")
            return examples

        for item in data["data"]:
            if "context" in item and "truth" in item:
                # Assuming 'truth' directly maps to the illness name for this example
                example = dspy.Example(
                    symptom_context=item["context"], extracted_illness=item["truth"]
                ).with_inputs("symptom_context")
                examples.append(example)
            else:
                logger.warning(f"Skipping malformed item in JSON data: {item}")
        logger.info(f"Loaded {len(examples)} examples from {json_path}")
    except FileNotFoundError:
        logger.error(f"Training data file not found: {json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
    return examples


# --- Metric Function ---
def simple_exact_match_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> bool:
    """A simple metric that checks for exact match of the extracted illness."""
    # Ensure prediction.extracted_illness is not None
    if prediction.extracted_illness is None:
        logger.warning(f"Prediction for example '{example.symptom_context}' is None.")
        return False

    # Normalize both for comparison (lowercase, strip whitespace)
    predicted_illness = str(prediction.extracted_illness).strip().lower()
    true_illness = str(example.extracted_illness).strip().lower()

    is_match = predicted_illness == true_illness
    if not is_match:
        logger.debug(
            f"Metric Mismatch: Expected '{true_illness}', Got '{predicted_illness}' for context '{example.symptom_context}'"
        )
    return is_match


# --- Main Optimization Script ---
def main():
    logger.info("Starting DSPy illness description optimization...")

    # --- Determine script's directory for relative paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")

    try:
        lm = dspy.LM(
            model="gemini/gemini-2.5-flash-preview-04-17", max_output_tokens=65_000
        )
        dspy.settings.configure(lm=lm)
        logger.info(
            f"DSPy configured with LM: {type(lm).__name__} using model {lm.kwargs.get('model')}"
        )
    except Exception as e:
        logger.error(
            f"Failed to configure language model: {e}. Please ensure your LM provider (e.g., OpenAI, Google) is correctly set up."
        )
        return

    # Load data
    train_data_path = os.path.join(script_dir, "data.json")
    logger.info(f"Attempting to load training data from: {train_data_path}")
    trainset = load_training_data(train_data_path)
    if not trainset:
        logger.error("No training data loaded. Exiting.")
        return

    # Split data if needed (e.g., for a separate dev/validation set for the optimizer)
    # For MIPROv2, it can internally split or use a validation set.
    # Let's use a simple split for demonstration if the dataset is large enough.
    if len(trainset) < 10:  # MIPRO needs some examples for train/val
        logger.warning(
            "Dataset is very small. Using all for training and validation if optimizer requires."
        )
        devset = trainset
    else:
        # A common split, but MIPROv2's `auto` or explicit `valset` handles this.
        # Let's assume we'll pass the full trainset and let MIPROv2 manage.
        # If you want to manually control validation set for MIPRO:
        # split_point = int(len(trainset) * 0.7)
        # optim_trainset = trainset[:split_point]
        # valset = trainset[split_point:]
        pass  # MIPRO can take the full trainset and sample from it or use a valset.

    # Get initial description from Pydantic schema
    # pydantic_schema = IllnessExtraction()
    initial_description = IllnessExtraction.model_fields["illness"].description
    if not initial_description:
        logger.error(
            "Description for 'illness' field in Pydantic schema is empty. Cannot optimize."
        )
        return
    logger.info(
        f"Initial Pydantic description for 'illness' field: '{initial_description}'"
    )

    # Initialize the DSPy module with the initial description
    # This module's signature will be the target for optimization.
    student_program = IllnessExtractorModule(initial_instruction=initial_description)

    # Configure the optimizer (MIPROv2 for instruction optimization)
    # We want zero-shot instruction optimization, so demos are set to 0.
    optimizer = dspy.MIPROv2(
        metric=simple_exact_match_metric,
        prompt_model=lm,  # Model used for proposing instructions
        task_model=lm,  # Model used for evaluating the program
        num_threads=10,
        max_bootstrapped_demos=0,  # Zero-shot for instruction only
        max_labeled_demos=0,  # Zero-shot for instruction only
        auto=None,
        num_candidates=5,  # Number of candidate programs/instructions to explore (adjust as needed)
        init_temperature=0.7,  # Temperature for instruction generation
        verbose=True,
        log_dir=os.path.join(
            script_dir, "optimise_logs"
        ),  # Directory to store optimizer logs relative to script
    )

    logger.info("Starting optimization process with MIPROv2...")
    # MIPROv2's compile method takes trainset. It can also take a valset.
    # If valset is not provided, it might sample from trainset.
    # The `auto` parameter in MIPROv2 (e.g., "light", "medium") also influences this.
    # Let's use `num_trials` for more explicit control when not using `auto`.
    # The `compile` method requires `requires_permission_to_run=False` for programmatic execution
    # or it will prompt the user.

    try:
        # For programmatic run without interactive confirmation:
        # optimized_program = optimizer.compile(student_program, trainset=trainset, num_trials=10, valset=devset, requires_permission_to_run=False)
        # If devset is small or not explicitly managed, MIPRO will handle validation split from trainset.
        optimized_program = optimizer.compile(
            student_program,
            trainset=trainset,
            num_trials=10,  # Adjust number of optimization trials
            requires_permission_to_run=False,  # Important for non-interactive runs
        )
    except Exception as e:
        logger.error(f"Error during MIPROv2 compilation: {e}")
        logger.error("Ensure your LM is configured correctly and has sufficient quota.")
        logger.error(f"Traceback: {e.__traceback__}")
        return

    logger.info("Optimization process completed.")

    # --- Extracting and Comparing Descriptions ---
    # The optimized instruction is within the signature of the predictor in the optimized module.
    original_instruction = student_program.predictor.signature.instructions

    # MIPROv2 replaces the student program's modules with optimized versions.
    # So, optimized_program.predictor should contain the optimized signature.
    optimized_instruction = optimized_program.predictor.signature.instructions

    logger.info(f"Original Instruction (from Pydantic):\n'{original_instruction}'")
    logger.info(f"Optimized Instruction (by DSPy):\n'{optimized_instruction}'")

    # You can now take `optimized_instruction` and update your Pydantic schema's
    # field description manually or programmatically.

    # --- Saving the Optimized Program ---
    optimized_program_path = os.path.join(
        script_dir, "optimized_illness_extractor.json"
    )
    try:
        optimized_program.save(optimized_program_path)
        logger.info(f"Optimized DSPy program saved to {optimized_program_path}")
    except Exception as e:
        logger.error(f"Failed to save optimized program: {e}")

    # --- (Optional) Evaluating the Optimized Program ---
    # You might want to evaluate on a separate test set if you have one.
    # For example:
    # testset = ... # load your test data
    # evaluate = dspy.Evaluate(devset=testset, metric=simple_exact_match_metric, num_threads=1, display_progress=True)
    # scores = evaluate(optimized_program)
    # logger.info(f"Evaluation score on test set: {scores}")


if __name__ == "__main__":
    main()
