# Data Visualization Crash Course

Matplotlib and pandas visualization  
  
**Imports**

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

In order to see data visualzations with the Jypter  
Known as a _"magic"_ command. Works in Juytper Notebook

```py
%matplotlib inline
```

**Create a Graph of Two Arrays**

```py
x = np.arange(0,10)
y = x**2
plt.plot(x,y)
```

![](/assets/VisualizationGraph0.png)

**Create a Graph with \* as plot points!**

```py
plt.plot(x,y, '*')
```

![](/assets/VI2.png)

**Create a Graph with a Red Line**

```py
plt.plot(x,y, 'red')
```

![](/assets/VI3.png)

**Create a Graph with a Red Dotted Line**

```py
plt.plot(x,y, 'r--')
```

![](/assets/VI4.png)

**Create a Graph with a Red Dotted Line and start at \(0,0\)**

```py
plt.plot(x,y, 'r--')
plt.xlim(0)
```

![](/assets/VI5.png)

**Create a Graph with a Red Dotted Line from x of \(0:4\)**

```py
plt.plot(x,y, 'r--')
plt.xlim(0,4)
```

![](/assets/VI6.png)

**Create a Graph with a Red Dotted Line from x of \(0:4\) and y \(0:10\)**

```py
plt.plot(x,y, 'r--')
plt.xlim(0,4)
plt.ylim(0,10)
```

![](/assets/VI7.png)

**Add a Title and Labels to a Graph**

```py
plt.plot(x,y, 'r--')
plt.xlim(0,4)
plt.ylim(0,10)
plt.title("Simple Example")
plt.xlabel("Time")
plt.ylabel("Dollars in Millions")
```

![](/assets/VI8.png)

**Create a Color Mapping Graph based on values in a Matrix**

```py
mat = np.arange(0,100).reshape(10,10)
plt.imshow(mat)
```

![](/assets/VI9.png)

**Create a "CoolWarm" Head Graph**

```py
mat = np.arange(0,100).reshape(10,10)
plt.imshow(mat,cmap='coolwarm')
```

![](/assets/VI10.png)

They're many more colors. Quick list of them all.

> Accent, Accent\_r, Blues, Blues\_r, BrBG, BrBG\_r, BuGn, BuGn\_r, BuPu, BuPu\_r, CMRmap, CMRmap\_r, Dark2, Dark2\_r, GnBu, GnBu\_r, Greens, Greens\_r, Greys, Greys\_r, OrRd, OrRd\_r, Oranges, Oranges\_r, PRGn, PRGn\_r, Paired, Paired\_r, Pastel1, Pastel1\_r, Pastel2, Pastel2\_r, PiYG, PiYG\_r, PuBu, PuBuGn, PuBuGn\_r, PuBu\_r, PuOr, PuOr\_r, PuRd, PuRd\_r, Purples, Purples\_r, RdBu, RdBu\_r, RdGy, RdGy\_r, RdPu, RdPu\_r, RdYlBu, RdYlBu\_r, RdYlGn, RdYlGn\_r, Reds, Reds\_r, Set1, Set1\_r, Set2, Set2\_r, Set3, Set3\_r, Spectral, Spectral\_r, Vega10, Vega10\_r, Vega20, Vega20\_r, Vega20b, Vega20b\_r, Vega20c, Vega20c\_r, Wistia, Wistia\_r, YlGn, YlGnBu, YlGnBu\_r, YlGn\_r, YlOrBr, YlOrBr\_r, YlOrRd, YlOrRd\_r, afmhot, afmhot\_r, autumn, autumn\_r, binary, binary\_r, bone, bone\_r, brg, brg\_r, bwr, bwr\_r, cool, cool\_r, coolwarm, coolwarm\_r, copper, copper\_r, cubehelix, cubehelix\_r, flag, flag\_r, gist\_earth, gist\_earth\_r, gist\_gray, gist\_gray\_r, gist\_heat, gist\_heat\_r, gist\_ncar, gist\_ncar\_r, gist\_rainbow, gist\_rainbow\_r, gist\_stern, gist\_stern\_r, gist\_yarg, gist\_yarg\_r, gnuplot, gnuplot2, gnuplot2\_r, gnuplot\_r, gray, gray\_r, hot, hot\_r, hsv, hsv\_r, inferno, inferno\_r, jet, jet\_r, magma, magma\_r, nipy\_spectral, nipy\_spectral\_r, ocean, ocean\_r, pink, pink\_r, plasma, plasma\_r, prism, prism\_r, rainbow, rainbow\_r, seismic, seismic\_r, spectral, spectral\_r, spring, spring\_r, summer, summer\_r, tab10, tab10\_r, tab20, tab20\_r, tab20b, tab20b\_r, tab20c, tab20c\_r, terrain, terrain\_r, viridis, viridis\_r, winter, winter\_r



