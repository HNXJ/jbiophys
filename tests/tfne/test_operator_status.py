from jbiophysic import jtfne
from jbiophysic.tfne.operator_status import get_operator_status


def test_operator_status_covers_formal_stack():
    status = get_operator_status()
    assert set(status) == {
        "emitter",
        "synapse",
        "chemical",
        "source_projection",
        "field",
        "probe",
        "analysis",
        "optimizer",
        "constraints",
    }
    assert status["chemical"].state == "specified_future_module"
    assert "q_chem" in status["chemical"].claim_forbidden


def test_jtfne_status_and_operator_graph():
    assert "source_projection" in jtfne.status()
    graph = jtfne.operator_graph()
    assert graph[0] == "E_theta"
    assert "F_Omega_B_G_Gamma" in graph
