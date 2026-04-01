from docuweave.models import Block, BlockType, Document, DocumentMetadata

block = Block(
    id="b1",
    type=BlockType.PARAGRAPH,
    text="Hello world",
    page=1
)

doc = Document(
    metadata=DocumentMetadata(source="sample.pdf"),
    blocks=[block]
)

print(doc.model_dump())