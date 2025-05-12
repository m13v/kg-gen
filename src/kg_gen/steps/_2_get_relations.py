from typing import List, Tuple, Literal, Optional, Any
import dspy
from pydantic import BaseModel, create_model, Field

# Base class for Relation, so we can refer to it before dynamic creation if needed,
# though in this refactor it's fully defined inside.
class BaseRelation(BaseModel):
    subject: Any
    predicate: Any
    object: Any

def get_relations(dspy_instance: Any, input_data: str, entities: List[str], is_conversation: bool = False, edge_labels: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    
    # Determine types for Pydantic model fields
    # Ensure entities and edge_labels are not empty for Literal, otherwise Pydantic raises error
    # Also, Literal requires at least one value.
    subject_type: Any = str
    object_type: Any = str
    predicate_type: Any = str

    # Prepare detailed descriptions for prompts based on whether lists are provided
    subject_object_constraint_desc = "Subject and object MUST be exact matches from the provided 'entities' list."
    predicate_constraint_desc = "The 'predicate' MUST be an exact match from the provided 'edge_labels' list."

    if entities and len(entities) > 0:
        # Filter out empty strings from entities if any, as Literal with empty string can be problematic
        valid_entities = [e for e in entities if e]
        if valid_entities:
            subject_type = Literal[tuple(valid_entities)]
            object_type = Literal[tuple(valid_entities)]
        else: # All entities were empty strings or list became empty
            subject_object_constraint_desc = "The 'entities' list was empty or contained only empty strings; subjects/objects will be strings."
    else: # entities list is None or empty
        subject_object_constraint_desc = "No 'entities' list provided; subjects/objects will be strings."

    if edge_labels and len(edge_labels) > 0:
        valid_edge_labels = [e for e in edge_labels if e]
        if valid_edge_labels:
            predicate_type = Literal[tuple(valid_edge_labels)]
        else: # All edge_labels were empty strings or list became empty
            predicate_constraint_desc = "The 'edge_labels' list was empty or contained only empty strings; predicates will be strings."
    else: # edge_labels list is None or empty
        predicate_constraint_desc = "No 'edge_labels' list provided; predicates will be strings."

    # Dynamically create the Relation Pydantic model
    # Using Field(..., description=...) to pass descriptions to the LLM via schema
    Relation = create_model(
        'Relation',
        subject=(subject_type, Field(..., description="The subject of the relation. " + subject_object_constraint_desc)),
        predicate=(predicate_type, Field(..., description="The predicate of the relation. " + predicate_constraint_desc)),
        object=(object_type, Field(..., description="The object of the relation. " + subject_object_constraint_desc)),
        __base__=BaseModel # Simpler base, docstring will be in signature
    )
    # Add a docstring to the dynamically created model for clarity if it's inspected
    Relation.__doc__ = "Knowledge graph subject-predicate-object triple with strict type validation based on provided entities and edge_labels."

    # Define DSPy Signatures *inside* this function, using the dynamic Relation model
    class ExtractTextRelations(dspy.Signature):
        """Extract subject-predicate-object triples from the source text.
        {subject_object_constraint_desc_main}
        {predicate_constraint_desc_main}
        This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
        Adhere strictly to the types and constraints specified for subject, predicate, and object in the output schema."""
        
        source_text: str = dspy.InputField()
        entities: list[str] = dspy.InputField(desc="List of entities that can be used for subject and object.")
        edge_labels: List[str] = dspy.InputField(desc="List of preferred predicate types. If provided, predicates MUST come from this list.")
        relations: List[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples. MUST follow all constraints.")

        # Dynamically update the docstring
        __doc__ = __doc__.format(
            subject_object_constraint_desc_main=subject_object_constraint_desc,
            predicate_constraint_desc_main=predicate_constraint_desc
        )


    class ExtractConversationRelations(dspy.Signature):
        """Extract subject-predicate-object triples from the conversation, including:
        1. Relations between concepts discussed
        2. Relations between speakers and concepts (e.g. user asks about X)
        3. Relations between speakers (e.g. assistant responds to user)
        {subject_object_constraint_desc_main}
        {predicate_constraint_desc_main}
        This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
        Adhere strictly to the types and constraints specified for subject, predicate, and object in the output schema."""
        
        source_text: str = dspy.InputField()
        entities: list[str] = dspy.InputField(desc="List of entities that can be used for subject and object.")
        edge_labels: List[str] = dspy.InputField(desc="List of preferred predicate types. If provided, predicates MUST come from this list.")
        relations: List[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples. MUST follow all constraints.")
        
        # Dynamically update the docstring
        __doc__ = __doc__.format(
            subject_object_constraint_desc_main=subject_object_constraint_desc,
            predicate_constraint_desc_main=predicate_constraint_desc
        )

    if is_conversation:
        extract = dspy_instance.Predict(ExtractConversationRelations)
    else:
        extract = dspy_instance.Predict(ExtractTextRelations)
        
    # The `extract` call will now use the DSPy instance passed to get_relations
    # DSPy will attempt to parse LLM output into List[Relation].
    # If Pydantic validation (due to Literal) fails for an item,
    # dspy.Predict might raise an error or return fewer items.
    # This replaces the manual conformance checks.
    try:
        prediction = extract(source_text=input_data, entities=entities, edge_labels=edge_labels if edge_labels is not None else [])
        predicted_relations = prediction.relations if hasattr(prediction, 'relations') and prediction.relations is not None else []
    except Exception as e:
        print(f"[kg_gen - get_relations] Error during dspy.Predict or parsing relations: {e}")
        predicted_relations = []

    all_relations_tuples: List[Tuple[str, str, str]] = []
    if predicted_relations:
        for rel_model_instance in predicted_relations:
            # Ensure rel_model_instance is a Pydantic model (or has .subject, .predicate, .object)
            if hasattr(rel_model_instance, 'subject') and hasattr(rel_model_instance, 'predicate') and hasattr(rel_model_instance, 'object'):
                 all_relations_tuples.append((rel_model_instance.subject, rel_model_instance.predicate, rel_model_instance.object))
            else:
                print(f"[kg_gen - get_relations] Warning: Received an item in relations list that is not a valid Relation model: {rel_model_instance}")


    # If verbose logging is needed for understanding Pydantic validation issues,
    # this is where one might inspect the LLM's raw output vs parsed output if DSPy allows.
    # For now, we rely on Pydantic raising errors or DSPy filtering them.

    return all_relations_tuples