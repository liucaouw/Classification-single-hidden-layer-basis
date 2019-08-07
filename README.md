# Classification using single hidden layer basis

When employng M single hidden layer basis features (using any activation a(.)) the full gradient of a cost g (e.g., the softmax) is a vector of length Q=M(N+2)+1 containing the derivation of the cost with respect to each variable.

$$\nabla g=\begin{bmatrix}
\frac{\partial }{\partial b} g& \frac{\partial }{\partial w_{1}}g & ... & \frac{\partial }{\partial w_{M}}g & \frac{\partial}{\partial c_{1}}g &  ... & \frac{\partial}{\partial c_{M}}g & \nabla_{\mathbf{v_{1}}}^{T}g & ...  & \nabla_{\mathbf{v_{M}}}^{T}g 
\end{bmatrix}^{T}$$ 

Where the derivatives are easily calculated using the chain rule. The derivatives of this gradient (using the softmax cost) are given by

$$\frac{\partial}{\partial b}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )y_{p}$$

$$\frac{\partial}{\partial w_{n}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )y_{p}$$

$$\frac{\partial}{\partial c_{n}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )w_{n}y_{p}$$

$$\nabla_{\mathbf{v_{n}}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )\mathbf{x_{p}}w_{n}y_{p}$$ 

### proof 

The cost function $ g\left ( \mathbf{w} \right )=\sum_{p=1}^{P}log\left ( 1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )} \right )$

where $$ \mathbf{f_{p}}=\begin{bmatrix}
a\left ( c_{1}+\mathbf{x_{p}^{T}}\mathbf{v_{1}} \right )& a\left ( c_{2}+\mathbf{x_{p}^{T}}\mathbf{v_{2}} \right ) & ... & a\left ( c_{M}+\mathbf{x_{p}^{T}}\mathbf{v_{M}} \right )
\end{bmatrix}^{T}$$

$$ \mathbf{w}=\begin{bmatrix}
w_{1} & w_{2} & ... & w_{M}
\end{bmatrix}$$

$$b=w_{0}$$

Thus, 

$$ \frac{\partial}{\partial b}g=\sum_{p=1}^{P}\frac{e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )=\sum_{p=1}^{P}\frac{1}{1+e^{y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )$$
$$ =-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )y_{p}$$

Thus,

$$ \frac{\partial}{\partial w_{n}}g=\sum_{p=1}^{P}\frac{e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot a\left ( c_{n}+ \mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )\cdot \left (-y_{p}  \right )$$
$$ =-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )y_{p}$$

Thus,

$$ \frac{\partial}{\partial c_{n}}g=\sum_{p=1}^{P}\frac{e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )\cdot w_{n}\cdot a'\left ( c_{n}+ \mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )$$
$$=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )w_{n}y_{p}$$

Thus,

$$ \nabla_{\mathbf{v_{n}}}g=\sum_{p=1}^{P}\frac{e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )\cdot w_{n}\cdot a'\left ( c_{n}+ \mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )\cdot \mathbf{x_{p}}$$
$$=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )\mathbf{x_{p}}w_{n}y_{p}$$


This gradient can be written more efficiently for Python that have especially good implementations of matrix/vector operations by writing it more compactly. Supposing that $ a=tanh\left ( \cdot  \right ) $ is the activation function (meaning $ a'=sech^{2}\left ( \cdot  \right ) $ is the hyperbolic secant function squared), the derivatives from above may be written more compactly as

$$ \frac{\partial}{\partial b}g=-\mathbf{1_{P\times 1}^{T} q}\odot \mathbf{y}$$

$$ \frac{\partial}{\partial w_{n}}g=-\mathbf{1_{P\times 1}^{T}}\left ( \mathbf{q}\odot \mathbf{t_{n}}\odot \mathbf{y} \right )$$

$$ \frac{\partial}{\partial c_{n}}g=-\mathbf{1_{P\times 1}^{T}}\left ( \mathbf{q}\odot \mathbf{s_{n}}\odot \mathbf{y} \right )w_{n}$$

$$ \nabla_{\mathbf{v_{n}}}g=-\mathbf{X}\cdot \mathbf{q}\odot \mathbf{s_{n}}\odot \mathbf{y}w_{n}$$ 


where $\bigodot $ denotes the component-wise product and denoting $ q_{p}=\sigma \left ( -y_{p}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{p}^{T}v_{m}}\right ) \right ) \right ) $, $ t_{np}=tanh\left ( c_{n}+\mathbf{x_{p}^{T}v_{n}} \right )$, $ s_{np}=sech^{2}\left ( c_{n}+\mathbf{x_{p}^{T}v_{n}} \right )$, and $ \mathbf{q}$,$ \mathbf{t_{n}}$, and $ \mathbf{s_{n}}$ the P length vectors containing these entries.

### proof 

Due to

$$\mathbf{q}=\begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}v_{m}} \right ) \right ) \right )\\ 
\sigma \left ( -y_{2}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{2}^{T}v_{m}} \right ) \right ) \right )
\\ 
...
\\ 
\sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}v_{m}} \right ) \right ) \right )
\end{bmatrix}^{T}$$

$$\mathbf{t_{n}}=\begin{bmatrix}
tanh\left ( c_{n}+\mathbf{x_{1}^{T}v_{n}} \right )\\ tanh\left ( c_{n}+\mathbf{x_{2}^{T}v_{n}} \right )
\\ ...
\\ tanh\left ( c_{n}+\mathbf{x_{P}^{T}v_{n}} \right )
\end{bmatrix}^{T}$$

$$\mathbf{s_{n}}=\begin{bmatrix}
sech^{2}\left ( c_{n}+\mathbf{x_{1}^{T}v_{n}} \right )\\ sech^{2}\left ( c_{n}+\mathbf{x_{2}^{T}v_{n}} \right )
\\ ...
\\ sech^{2}\left ( c_{n}+\mathbf{x_{P}^{T}v_{n}} \right )
\end{bmatrix}^{T}$$ 


$$\mathbf{y}=\begin{bmatrix} y_{1} &y_{2}  & ... & y_{P}\end{bmatrix}^{T}$$

Thus,

$$\frac{\partial}{\partial b}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )y_{p}$$ 


$$ =  - \begin{bmatrix} 1 &1  &...  &1 \end{bmatrix}_{1 \times P}$$

$$\cdot \begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}v_{m}} \right ) \right ) \right )\cdot y_{1} &...  &\sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}v_{m}} \right ) \right ) \right )\cdot y_{P} \end{bmatrix}^{T}$$

$$ =-\mathbf{1_{P\times 1}^{T} q}\odot \mathbf{y}$$

where $tanh\left ( c_{m}+\mathbf{x_{P}^{T}v_{m}} \right )=a\left (  c_{m}+\mathbf{x_{P}^{T}v_{m}}\right )$. 


Thus,

$$ \frac{\partial}{\partial w_{n}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )y_{p}$$

$$ =  - \begin{bmatrix} 1 &1  &...  &1 \end{bmatrix}_{1 \times P}$$

$$ \cdot \begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot tanh\left ( c_{n}+\mathbf{x_{1}^{T}}\mathbf{v_{n}} \right )\cdot y_{1} &...  & \sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot tanh\left ( c_{n}+\mathbf{x_{P}^{T}}\mathbf{v_{n}} \right ) \cdot y_{P}
\end{bmatrix}^{T}$$


$$ =-\mathbf{1_{P\times 1}^{T}}\left ( \mathbf{q}\odot \mathbf{t_{n}}\odot\mathbf{y} \right )$$

