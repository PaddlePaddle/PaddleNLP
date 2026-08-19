# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

# Create test application and route directly matching file_upload controller logic
app = FastAPI()

TEMP_PARSE_DIR = tempfile.mkdtemp(prefix="paddlenlp_test_parse_")
FILE_PARSE_PATH = TEMP_PARSE_DIR

# Place a sample parsed file
SAMPLE_FILENAME = "1fc0aeac9900487a8c6cec8dda6499bd_demo_1.png"
SAMPLE_CONTENT = b"PNG_IMAGE_SAMPLE_DATA_12345"
with open(os.path.join(FILE_PARSE_PATH, SAMPLE_FILENAME), "wb") as f:
    f.write(SAMPLE_CONTENT)


@app.get("/files")
def download_file(file_name: str = "1fc0aeac9900487a8c6cec8dda6499bd_demo_1.png"):
    if not file_name or file_name in (".", "..") or "/" in file_name or "\\" in file_name or ".." in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    file_name = os.path.basename(file_name)
    if not file_name or file_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid file name")

    abs_base = os.path.abspath(FILE_PARSE_PATH)
    abs_target = os.path.abspath(os.path.join(FILE_PARSE_PATH, file_name))
    if not abs_target.startswith(abs_base + os.sep):
        raise HTTPException(status_code=400, detail="Invalid file name")

    if os.path.exists(abs_target) and os.path.isfile(abs_target):
        return FileResponse(abs_target)
    return {"message": "File not Found"}


@pytest.fixture
def client():
    return TestClient(app)


def test_download_file_valid(client: TestClient):
    response = client.get(f"/files?file_name={SAMPLE_FILENAME}")
    assert response.status_code == 200
    assert response.content == SAMPLE_CONTENT


def test_download_file_default(client: TestClient):
    response = client.get("/files")
    assert response.status_code == 200
    assert response.content == SAMPLE_CONTENT


def test_download_file_non_existent(client: TestClient):
    response = client.get("/files?file_name=non_existent_file_98765.png")
    assert response.status_code == 200
    assert response.json() == {"message": "File not Found"}


def test_download_file_absolute_path_traversal(client: TestClient):
    # Test POSIX and Windows absolute path traversal attempts
    for abs_path in ["/etc/passwd", "/etc/hosts", "C:\\Windows\\win.ini", "C:\\Windows\\System32\\drivers\\etc\\hosts"]:
        response = client.get(f"/files?file_name={abs_path}")
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid file name"}


def test_download_file_relative_path_traversal(client: TestClient):
    # Test relative directory traversal attempts
    for traversal_path in ["../../etc/passwd", "../../../secret.txt", "..\\..\\Windows\\win.ini"]:
        response = client.get(f"/files?file_name={traversal_path}")
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid file name"}


def test_download_file_dot_or_directory(client: TestClient):
    # Test empty or directory navigation tokens
    for invalid_name in [".", "..", "/"]:
        response = client.get(f"/files?file_name={invalid_name}")
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid file name"}
