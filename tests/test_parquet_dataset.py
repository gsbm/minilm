import pyarrow as pa
import pyarrow.parquet as pq

from minilm import Tokenizer, Trainer


def test_load_parquet_dataset(tmp_path):
    records = [
        {"text": "first example"},
        {"text": "second example"},
        {"text": "third example"},
    ]
    table = pa.Table.from_pylist(records)
    parquet_path = tmp_path / "dataset.parquet"
    pq.write_table(table, parquet_path)

    trainer = Trainer()
    dataset = trainer.load_dataset(parquet_path, file_format="parquet")

    assert dataset == [row["text"] for row in records]

    tokenizer = Tokenizer(vocab_size=64)
    tokenizer.fit(dataset)
    encoded = tokenizer.encode(dataset[0])
    assert isinstance(encoded, list) and encoded


def test_load_parquet_directory(tmp_path):
    records_part1 = [{"text": "chunk one"}, {"text": "chunk two"}]
    records_part2 = [{"text": "chunk three"}]
    dir_path = tmp_path / "sharded"
    dir_path.mkdir()

    pq.write_table(pa.Table.from_pylist(records_part1), dir_path / "part-0.parquet")
    pq.write_table(pa.Table.from_pylist(records_part2), dir_path / "part-1.parquet")

    trainer = Trainer()
    dataset = trainer.load_dataset(dir_path, file_format="parquet")

    assert dataset == ["chunk one", "chunk two", "chunk three"]
