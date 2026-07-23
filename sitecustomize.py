import sys
import types


gym_notices = types.ModuleType("gym_notices")
gym_notices_notices = types.ModuleType("gym_notices.notices")
gym_notices_notices.notices = {}
gym_notices.notices = gym_notices_notices
sys.modules.setdefault("gym_notices", gym_notices)
sys.modules.setdefault("gym_notices.notices", gym_notices_notices)
