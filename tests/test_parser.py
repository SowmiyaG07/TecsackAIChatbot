from app.ingestion.document_parser import parse_document

file_path = "data/hr/Holidaylist2026.pdf"

elements = parse_document(file_path)

print()
print("=" * 80)
print("PARSER TEST")
print("=" * 80)

print(f"Elements: {len(elements)}")

for index, element in enumerate(elements):

    print()
    print(f"[{index}]")
    print(f"TYPE: {element.element_type}")

    method = (
        element.metadata.get("extraction_method")
        if element.metadata
        else None
    )

    print(f"METHOD: {method}")
    print(f"PAGE: {element.page}")

    print("TEXT:")
    print(element.text)

    print("-" * 60)