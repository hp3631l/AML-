import os
import sys
import copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from security.secure_cleanup import secure_wipe_inplace
from bank_node.api import app, get_embedding

def test_secure_wipe_inplace_list():
    """Test 1: verify secure_wipe_inplace([1,2,3]) leaves the list empty"""
    data = [1, 2, 3]
    secure_wipe_inplace(data)
    assert len(data) == 0, "List was not wiped"

def test_secure_wipe_inplace_dict_copy():
    """Test 2: verify that if you copy a dict first, then wipe the original, the copy is unaffected"""
    original = {"a": 1, "b": 2, "c": 3}
    copied = copy.deepcopy(original)
    
    secure_wipe_inplace(original)
    
    assert len(original) == 0, "Original dict was not wiped"
    assert len(copied) == 3, "Copied dict was affected"
    assert copied["a"] == 1, "Copied data changed"

from unittest.mock import patch

def test_bank_api_embedding_non_empty():
    """Test 3: verify the bank API embedding endpoint returns non-empty JSON"""
    mock_embedding = {
        "account_id": "ACC-TEST-123",
        "bank_id": "bank_a",
        "profile_vector": [0.1] * 34,
        "occupation_code": 1,
        "salary_band": 3,
        "country_code": "US",
        "mismatch_score": 0.05
    }
    
    with patch("bank_node.api.generate_embedding", return_value=mock_embedding):
        with patch("bank_node.api._DB_CONN", new="MOCKED_CONN"):
            with patch("bank_node.api._BANK_ID", new="bank_a"):
                response = get_embedding("ACC-TEST-123")
                
                assert isinstance(response, dict), "Response should be a dictionary"
                assert response.get("account_id") == "ACC-TEST-123", "Account ID mismatch"
                
                # Check it hasn't been wiped
                resp_data = response.dict() if hasattr(response, "dict") else response
                assert isinstance(resp_data, dict), "Response is not a dict or pydantic model"
                assert len(resp_data) > 0, "Response JSON is empty!"

if __name__ == "__main__":
    test_secure_wipe_inplace_list()
    test_secure_wipe_inplace_dict_copy()
    test_bank_api_embedding_non_empty()
    print("All tests passed successfully.")
