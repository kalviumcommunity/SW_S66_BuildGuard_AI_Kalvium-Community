from types import SimpleNamespace

from src.embedding_demo import (
    compare_embeddings,
    cosine_similarity,
    dot_product,
    run_embedding_demo,
    validate_embedding_dimensions,
)


def raises_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_cosine_identical_and_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_dot_product_and_dimension_errors() -> None:
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0
    raises_value_error(dot_product, [1.0], [1.0, 2.0])
    raises_value_error(cosine_similarity, [0.0, 0.0], [1.0, 0.0])


def test_embedding_dimensions() -> None:
    assert validate_embedding_dimensions([[1.0, 0.0], [0.0, 1.0]]) == (True, 2)
    raises_value_error(validate_embedding_dimensions, [[1.0], [1.0, 2.0]])


def test_comparison_reports_similar_score_higher_with_dot_product() -> None:
    result = compare_embeddings([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert result["similarity_metric"] == "dot_product"
    assert result["similar_score_higher"] is True
    assert float(result["similarity_similar_leave_pair"]) > float(result["similarity_unrelated_pair"])


def test_injected_model_result_is_deterministic_without_loading_model() -> None:
    fake_model = SimpleNamespace(
        embed_documents=lambda texts: [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    )
    result = run_embedding_demo(
        fake_model,
        ["leave one", "leave two", "cafeteria"],
        model_name="fake-model",
        write_outputs=False,
    )
    assert result["model"] == "fake-model"
    assert result["vector_dimension"] == 2
    assert result["same_length"] is True
    assert result["comparison_result"] == "PASS"
