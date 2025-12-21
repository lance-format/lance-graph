# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Integration tests for LanceGraphStore with GCS.

NOTE: This test currently fails because the underlying Rust `lance` implementation
may not be fully respecting the `google_base_url` or `STORAGE_EMULATOR_HOST`
when accessed via the Python bindings in this environment. It defaults to
`storage.googleapis.com` which causes 403/401 errors with mock credentials.

To Debug:
1. Ensure `lance` python package version supports `storage_options` propagation for GCS.
2. Check if `object_store` crate in Rust needs specific flags for emulator support.
3. Verify if `disable_oauth` is correctly parsed by the Rust layer.
"""

import json
import logging
import os

import pyarrow as pa
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from knowledge_graph.config import KnowledgeGraphConfig
from knowledge_graph.store import LanceGraphStore

# Configure logging to see more details if possible
logging.basicConfig(level=logging.DEBUG)

# @pytest.fixture(scope="module")
# def gcs_server():
#     """Start a Fake GCS Server container."""
#     # Using fake-gcs-server which provides a GCS-compatible API
#     container = DockerContainer("fsouza/fake-gcs-server:latest")
#     container.with_exposed_ports(4443)
#     container.with_command(
#         "-scheme http -external-url http://localhost:4443"
#     )  # Use HTTP for simplicity

#     container.start()

#     host_ip = container.get_container_host_ip()
#     port = container.get_exposed_port(4443)
#     url = f"http://{host_ip}:{port}"

#     # Wait for the server to be ready
#     start_time = time.time()
#     while time.time() - start_time < 10:
#         try:
#             requests.get(f"{url}/storage/v1/b", timeout=1)
#             break
#         except requests.exceptions.ConnectionError:
#             time.sleep(0.5)
#     else:
#         container.stop()
#         pytest.fail("Fake GCS Server failed to start")

#     yield url
#     container.stop()


@pytest.fixture
def fake_creds(tmp_path):
    """Create a fake GCS service account file with a valid RSA key structure."""
    # 1. Generate a real 2048-bit RSA private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # 2. Serialize it to PEM format (which GCS expects in the JSON)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    # 3. Build the Service Account JSON structure
    creds = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test_key_id",
        "private_key": private_key_pem,
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
    }

    path = tmp_path / "fake_gcs_creds.json"
    path.write_text(json.dumps(creds))
    return str(path)


@pytest.fixture
def gcs_store(fake_creds, monkeypatch):
    """Create a bucket and return a LanceGraphStore configured for it."""
    # Create a bucket
    bucket_name = "lance-poc"
    # # Ensure server is up and bucket creation works (equivalent to curl check)
    # resp = requests.post(
    #     f"{gcs_server}/storage/v1/b",
    #     params={"project": "test-project"},
    #     json={"name": bucket_name},
    #     headers={"Content-Type": "application/json"}
    # )

    # if resp.status_code not in (200, 201, 409):
    #     # 200/201 Created, 409 Conflict (exists)
    #     pytest.fail(
    #         f"Failed to verify Fake GCS is working. "
    #         f"Status: {resp.status_code}, Body: {resp.text}"
    #     )

    # print(f"Verified Fake GCS bucket creation: {resp.status_code}")

    # # Set up environment variables to force GCS emulator usage
    # host_no_scheme = gcs_server.replace("http://", "")

    # monkeypatch.setenv("STORAGE_EMULATOR_HOST", host_no_scheme)

    # monkeypatch.setenv(
    #     "GOOGLE_APPLICATION_CREDENTIALS", "/home/user/oss/credential.json"
    # )
    # monkeypatch.setenv("GOOGLE_ENDPOINT_URL", gcs_server)
    # monkeypatch.setenv("STORAGE_ALLOW_HTTP", "true")
    # monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT", fake_creds)
    # monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    config = KnowledgeGraphConfig(
        storage_path=f"gs://{bucket_name}/graph",
        schema_path=f"gs://{bucket_name}/graph/graph.yaml",
        storage_options={
            # "google_endpoint_url": gcs_server,
            # "endpoint": gcs_server,
            "token": "/home/user/oss/credential.json",
            "google_service_account": "/home/user/oss/credential.json",
            # "allow_http": "true",
            # "disable_oauth": "true",
            # "allow_invalid_certificates": "true",
        },
    )

    return LanceGraphStore(config)


@pytest.mark.skipif(
    os.environ.get("RUN_GCS_INTEGRATION") != "true",
    reason="Skipping GCS integration test; set RUN_GCS_INTEGRATION=true to run",
)
@pytest.mark.integration
def test_gcs_integration_write_read(gcs_store):
    """Test writing and reading from Fake GCS."""

    # Create a sample table
    table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})

    # Write to GCS
    # This calls lance.write_dataset internally
    gcs_store.write_tables({"Person": table})

    # Check if dataset exists via list_datasets
    # list_datasets uses pyarrow.fs.FileSystem.from_uri
    # pyarrow should respect STORAGE_EMULATOR_HOST env var we set in fixture
    datasets = gcs_store.list_datasets()
    assert "Person" in datasets
    assert "gs://lance-poc/graph/Person.lance" in datasets["Person"]

    # Read back
    # This calls lance.dataset internally
    loaded = gcs_store.load_tables(["Person"])
    assert "Person" in loaded
    assert loaded["Person"].to_pydict() == table.to_pydict()
