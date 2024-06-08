import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .prompts import get_condensed_question_prompt, get_qa_prompt
