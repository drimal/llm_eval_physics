import pytest
import os
import json
import tempfile
import pytest
from llm_eval_physics.data_loader import read_jsonl

@pytest.fixture
def sample_jsonl_file():
    test_data = [
        '{"key" : "value1"}\n',
        '{"key" : "value2"}\n',
        "null\n",
        '{"invalid_json"\n'  # Invalid JSON to test exception handling
    ]
    
    temp_dir = tempfile.mkdtemp()
    tmp_filename = os.path.join(temp_dir, 'test_data.jsonl')
    
    try:
        with open(tmp_filename, mode='w', encoding='utf-8') as tmp_file:
            tmp_file.writelines(test_data)
        
        yield tmp_filename
        
    finally:
        os.remove(tmp_filename)
        os.rmdir(temp_dir)

def test_read_valid_jsonl(sample_jsonl_file):
    try:
        results = read_jsonl(sample_jsonl_file)
        print("Actual Results:", results)  # Debug print
        expected_results = [
            {"key": "value1"},
            {"key": "value2"},
            None,  # Expecting None for 'null' line
            # Add more expected results based on your function's behavior
        ]
        assert results == expected_results
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")

def test_read_invalid_jsonl(sample_jsonl_file):
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(sample_jsonl_file)