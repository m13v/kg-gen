from typing import List, Optional
import dspy 

class TextEntities(dspy.Signature):
  """Extract key entities from the source text. Extracted entities are subjects or objects.
  If a list of `node_labels` (preferred entity types/categories) is provided, consider these types when identifying entities or try to classify extracted entities under these types.
  This is for an extraction task, please be THOROUGH and accurate to the reference text."""
  
  source_text: str = dspy.InputField()
  node_labels: List[str] = dspy.InputField(desc="Optional list of preferred entity types or categories. If provided, consider these when extracting entities.")
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class ConversationEntities(dspy.Signature):
  """Extract key entities from the conversation Extracted entities are subjects or objects.
  Consider both explicit entities and participants in the conversation.
  If a list of `node_labels` (preferred entity types/categories) is provided, consider these types when identifying entities or try to classify extracted entities under these types.
  This is for an extraction task, please be THOROUGH and accurate."""
  
  source_text: str = dspy.InputField()
  node_labels: List[str] = dspy.InputField(desc="Optional list of preferred entity types or categories. If provided, consider these when extracting entities.")
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

def get_entities(dspy: dspy.dspy, input_data: str, is_conversation: bool = False, node_labels: Optional[List[str]] = None) -> List[str]:
  if is_conversation:
    extract = dspy.Predict(ConversationEntities)
  else:
    extract = dspy.Predict(TextEntities)
    # print("input_data", input_data, "extract", extract)
    
  result = extract(source_text=input_data, node_labels=node_labels if node_labels is not None else [])
  return result.entities

