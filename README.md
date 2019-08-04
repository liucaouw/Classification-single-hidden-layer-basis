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

Thus, $ \frac{\partial}{\partial b}g=\sum_{p=1}^{P}\frac{e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )\\=\sum_{p=1}^{P}\frac{1}{1+e^{-y_{p}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} \right )}}\cdot \left (-y_{p}  \right )\\=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )y_{p}$
