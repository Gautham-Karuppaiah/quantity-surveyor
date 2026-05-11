import sqlite3
from main import AppController, ImportDrawing, LoadProject, Project, init_db


def test_round_trip(qtbot, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.qsproj"))
    init_db(conn)
    controller = AppController(Project(), conn)
    controller.dispatch(ImportDrawing("test.pdf"))
    conn.close()

    conn2 = sqlite3.connect(str(tmp_path / "test.qsproj"))
    project2 = Project()
    AppController(project2, conn2).dispatch(LoadProject())
    assert project2.active_page is not None
    assert not project2.active_page.pixmap.isNull()
    conn2.close()
