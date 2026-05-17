from src.eval.scorer import ExtractionScorer, RetrievalScorer


def test_mrr_perfect():
    scorer = RetrievalScorer()
    results = [["doc1", "doc2", "doc3"]]
    ground_truth = [["doc1"]]
    assert scorer.mrr(results, ground_truth) == 1.0


def test_mrr_second_position():
    scorer = RetrievalScorer()
    results = [["doc2", "doc1", "doc3"]]
    ground_truth = [["doc1"]]
    assert scorer.mrr(results, ground_truth) == 0.5


def test_mrr_not_found():
    scorer = RetrievalScorer()
    results = [["doc2", "doc3"]]
    ground_truth = [["doc1"]]
    assert scorer.mrr(results, ground_truth) == 0.0


def test_entity_f1_perfect():
    scorer = ExtractionScorer()
    p, r, f1 = scorer.entity_f1(["Apple", "Revenue"], ["Apple", "Revenue"])
    assert f1 == 1.0


def test_entity_f1_partial():
    scorer = ExtractionScorer()
    p, r, f1 = scorer.entity_f1(["Apple", "Google"], ["Apple", "Revenue"])
    assert 0 < f1 < 1.0
    assert p == 0.5
    assert r == 0.5


def test_entity_f1_empty():
    scorer = ExtractionScorer()
    p, r, f1 = scorer.entity_f1([], [])
    assert f1 == 1.0


def test_metric_accuracy():
    scorer = ExtractionScorer()
    pred = {"revenue": "$1,234 million", "eps": "$2.50"}
    expected = {"revenue": "$1234 million", "eps": "$2.50"}
    acc = scorer.metric_accuracy(pred, expected)
    assert acc == 1.0
