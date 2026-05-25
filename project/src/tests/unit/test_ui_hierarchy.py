"""Unit tests for XML-based sidebar state detection."""

from privacy_audit.utils.ui_hierarchy import analyze_sidebar_state

DRAWER_OPEN_XML = """
<hierarchy>
  <node class="androidx.drawerlayout.widget.DrawerLayout" bounds="[0,0][1080,1920]">
    <node class="android.widget.TextView" text="设置" clickable="true"
          displayed="true" bounds="[40,200][400,280]"/>
    <node class="android.widget.TextView" text="账号与安全" clickable="true"
          displayed="true" bounds="[40,300][400,380]"/>
  </node>
</hierarchy>
"""

TRIGGER_ONLY_XML = """
<hierarchy>
  <node class="android.widget.ImageButton" clickable="true" displayed="true"
        content-desc="更多" bounds="[20,80][120,180]"/>
  <node class="android.widget.TextView" text="我的" displayed="true"
        bounds="[400,1700][680,1850]"/>
</hierarchy>
"""


def test_drawer_open_detects_menu_entries() -> None:
    state = analyze_sidebar_state(DRAWER_OPEN_XML, screen_width=1080, screen_height=1920)
    assert state.drawer_open is True
    assert "设置" in state.menu_entries_visible
    assert state.sidebar_step_satisfied is True


def test_trigger_only_not_drawer_open() -> None:
    state = analyze_sidebar_state(TRIGGER_ONLY_XML, screen_width=1080, screen_height=1920)
    assert state.trigger_present is True
    assert state.drawer_open is False


KEEP_OPEN_XML = """
<hierarchy>
  <node class="androidx.drawerlayout.widget.DrawerLayout" bounds="[0,0][1080,1920]">
    <node resource-id="com.gotokeep.keep:id/viewToolbarSidebar"
          content-desc="打开侧边栏" clickable="true" displayed="true"
          class="android.widget.ImageView" bounds="[18,60][88,130]"/>
    <node resource-id="com.gotokeep.keep:id/view_main_slide"
          class="android.view.ViewGroup" displayed="true" bounds="[0,0][940,1920]">
      <node resource-id="com.gotokeep.keep:id/listItems"
            class="androidx.recyclerview.widget.RecyclerView"
            displayed="true" bounds="[28,70][912,1920]">
        <node class="android.view.ViewGroup" clickable="true" displayed="true"
              bounds="[56,1072][884,1118]">
          <node class="android.widget.TextView" text="设置"
                resource-id="com.gotokeep.keep:id/textName"
                displayed="true" bounds="[123,1076][842,1114]"/>
        </node>
      </node>
    </node>
  </node>
</hierarchy>
"""

KEEP_CLOSED_XML = """
<hierarchy>
  <node class="androidx.drawerlayout.widget.DrawerLayout" bounds="[0,0][1080,1920]">
    <node resource-id="com.gotokeep.keep:id/viewToolbarSidebar"
          content-desc="打开侧边栏" clickable="true" displayed="true"
          class="android.widget.ImageView" bounds="[18,60][88,130]"/>
  </node>
</hierarchy>
"""


def test_keep_open_drawer_from_capture_shape() -> None:
    state = analyze_sidebar_state(KEEP_OPEN_XML, screen_width=1080, screen_height=1920)
    assert state.drawer_open is True
    assert "设置" in state.menu_entries_visible
    assert state.trigger_present is False
    assert state.sidebar_step_satisfied is True


def test_keep_closed_from_capture_shape() -> None:
    state = analyze_sidebar_state(KEEP_CLOSED_XML, screen_width=1080, screen_height=1920)
    assert state.drawer_open is False
    assert state.trigger_present is True
