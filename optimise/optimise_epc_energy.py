import json
import logging
import os
from enum import Enum
from typing import List

import dspy
import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field

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


class EpcType(str, Enum):
    CONSUMPTION_CERTIFICATE = "CONSUMPTION_CERTIFICATE"
    DEMAND_CERTIFICATE = "DEMAND_CERTIFICATE"
    UNKNOWN_EPC_TYPE = "UNKNOWN_EPC_TYPE"

    def __str__(self):
        return self.value


class EPCFinalEnergyDemandExtraction(BaseModel):
    """Schema for extracting epc final energy demand information."""

    final_energy_demand_of_building: float = Field(
        description="""
            Endenergiebedarf.
            This refers to the final energy demand of the building.
            Do not extract the (Endenergieverbrauch) value of final energy consumption of the building.
            If the value is not present, return 0.0.
        """
    )


# --- DSPy Signature Definition ---
class EPCFinalEnergyDemandSignature(dspy.Signature):
    """
    Given a EPC document as list of images, extract the final energy demand of the building.
    """

    epc_context: List[dspy.Image] = dspy.InputField(
        desc="""
        The context of the energy performance certificate (EPC).
        This includes the property address, the energy performance rating, and any other relevant information.
        """
    )
    final_energy_demand_of_building: float = dspy.OutputField(
        desc="""
            Endenergiebedarf.
            This refers to the final energy demand of the building.
            Do not extract the (Endenergieverbrauch) value of final energy consumption of the building.
            If the value is not present, return 0.0.
        """
    )

    # The initial instruction for the 'final_energy_demand_of_building' field will be dynamically set
    # from the Pydantic schema's description during module initialization.


# --- DSPy Module ---
class EPCFinalEnergyDemandExtractorModule(dspy.Module):
    def __init__(self, initial_instruction: str):
        super().__init__()
        # We create a Predict module using a copy of the signature
        # and then update its instruction.
        self.predictor = dspy.Predict(EPCFinalEnergyDemandSignature)

        # Get the current signature instance from the predictor
        current_signature_instance = self.predictor.signature

        # Create a new signature instance with the updated instruction
        # using the .with_instructions() method.
        self.predictor.signature = current_signature_instance.with_instructions(
            initial_instruction
        )

        logger.info(
            f"Initial instruction for EPCFinalEnergyDemandExtractorModule set to: {self.predictor.signature.instructions}"
        )

    def forward(self, epc_context: List[dspy.Image], file_name: str) -> dspy.Prediction:
        return self.predictor(epc_context=epc_context)


# --- Data Loading Function ---
def load_training_data(
    ground_truth_path: str, epc_documents_path: str
) -> List[dspy.Example]:
    """
    Loads training data from a ground truth JSON file and corresponding PDF documents.

    Args:
        ground_truth_path: Path to the ground truth JSON file
        epc_documents_path: Path to the directory containing PDF files

    Returns:
        List of dspy.Example objects
    """
    examples = []
    try:
        # Load the ground truth JSON
        with open(ground_truth_path, "r") as f:
            data = json.load(f)

        # Check if the data has the expected structure
        if "datasetItems" not in data:
            logger.error("JSON data must contain a 'datasetItems' key")
            return examples

        # Process each item in the dataset
        for item in data["datasetItems"]:
            file_name = item.get("entityId")
            if not file_name or "final_energy_demand_of_building" not in item.get(
                "truth", {}
            ):
                logger.warning(f"Skipping item with missing data: {item}")
                continue

            # Find the corresponding PDF file
            pdf_path = os.path.join(epc_documents_path, f"{file_name}.pdf")
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue

            try:
                # Convert all PDF pages to PIL images
                pdf_document = fitz.open(pdf_path)
                epc_document_as_images = []

                # Process each page in the PDF
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap()

                    # Convert to PIL Image
                    img_data = pix.samples
                    img_size = (pix.width, pix.height)
                    pil_image = Image.frombytes("RGB", img_size, img_data)

                    # Convert to dspy.Image
                    dspy_image = dspy.Image.from_PIL(pil_image)
                    epc_document_as_images.append(dspy_image)

                # Create Example with all pages
                example = dspy.Example(
                    epc_context=epc_document_as_images,
                    file_name=file_name,
                    final_energy_demand_of_building=item.get("truth", {}).get(
                        "final_energy_demand_of_building"
                    ),
                ).with_inputs("epc_context", "file_name")

                examples.append(example)

            except Exception as e:
                logger.error(f"Error processing PDF {file_name}: {e}")

        logger.info(f"Loaded {len(examples)} examples from {ground_truth_path}")

    except FileNotFoundError:
        logger.error(f"Ground truth file not found: {ground_truth_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {ground_truth_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")

    return examples


# --- Metric Function ---
def simple_exact_match_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> bool:
    """A simple metric that checks for exact match of the extracted final energy demand of the building."""
    # Ensure prediction.final_energy_demand_of_building is not None
    if prediction.final_energy_demand_of_building is None:
        logger.warning(f"Prediction for example '{example.file_name}' is None.")
        return False

    # Normalize both for comparison (lowercase, strip whitespace)
    predicted_val = prediction.final_energy_demand_of_building
    true_val = example.final_energy_demand_of_building
    is_match = False
    if true_val is None:
        if float(predicted_val) == float(0.0):
            is_match = True
        else:
            is_match = False
    elif true_val is not None and float(predicted_val) == float(true_val):
        is_match = True

    logger.info(
        f"Original Predicted value: {predicted_val} for file {example.file_name}"
    )

    logger.info(
        f"Comparing predicted {predicted_val} with true {true_val} for file {example.file_name}"
    )
    if not is_match:
        logger.info(
            f"Metric Mismatch: Expected '{true_val}', Got '{predicted_val}' for context '{example.file_name}'"
        )
    return is_match


# --- Main Optimization Script ---
def main():
    logger.info("Starting DSPy epc description optimization...")

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
    ground_truth_path = os.path.join(
        script_dir,
        "documents/ground_truth/epc_de_test_dataset_49_geocoded.json",
    )
    epc_documents_path = os.path.join(
        script_dir,
        "documents/epc_files",
    )
    logger.info(f"Attempting to load training data from: {ground_truth_path}")
    trainset = load_training_data(ground_truth_path, epc_documents_path)
    if not trainset:
        logger.error("No training data loaded. Exiting.")
        return

    # Split data if needed (e.g., for a separate dev/validation set for the optimizer)
    # For MIPROv2, it can internally split or use a validation set.
    # Let's use a simple split for demonstration if the dataset is large enough.
    if len(trainset) < 0:  # MIPRO needs some examples for train/val
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

    initial_description = EPCFinalEnergyDemandExtraction.model_json_schema()
    initial_description = initial_description["properties"][
        "final_energy_demand_of_building"
    ]["description"]
    if not initial_description:
        logger.error(
            "Description for 'final_energy_demand_of_building' field in Pydantic schema is empty. Cannot optimize."
        )
        return
    logger.info(
        f"Initial Pydantic description for 'final_energy_demand_of_building' field: {initial_description}"
    )

    # Initialize the DSPy module with the initial description
    # This module's signature will be the target for optimization.
    student_program = EPCFinalEnergyDemandExtractorModule(
        initial_instruction=initial_description
    )

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
            script_dir, "optimise_logs_epc_final_energy_demand"
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
            num_trials=5,  # Adjust number of optimization trials
            requires_permission_to_run=False,  # Important for non-interactive runs
            # minibatch=True,
            # minibatch_size=10,
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
        script_dir, "optimized_epc_final_energy_demand_extractor.json"
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
    # evaluate = dspy.Evaluate(
    #     devset=testset,
    #     metric=simple_exact_match_metric,
    #     num_threads=1,
    #     display_progress=True,
    # )
    # scores = evaluate(optimized_program)
    # logger.info(f"Evaluation score on test set: {scores}")


if __name__ == "__main__":
    main()
