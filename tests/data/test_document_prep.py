from data_ingestion.data.document_prep import convert_pdf_to_markdown


def test_foo():
    assert convert_pdf_to_markdown("foo") == "foo"
