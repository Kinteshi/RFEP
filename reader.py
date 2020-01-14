# %%
import pstats
from pstats import SortKey
# %%
p = pstats.Stats('profile.profile')
p.strip_dirs()  # .sort_stats(-1).print_stats()

# %%
# p.sort_stats(SortKey.NAME)
# p.print_stats()
p.sort_stats(SortKey.CUMULATIVE, SortKey.TIME).print_stats('numpy')
#p.print_callers(.5, 'predict')
    