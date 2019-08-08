# Classifier using single hidden layer basis and k-nearest neighbors (k-NN) classifier

## single hidden layer basis

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

End.

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

where $tanh\left ( c_{m}+\mathbf{x_{p}^{T}v_{m}} \right )=a\left (  c_{m}+\mathbf{x_{p}^{T}v_{m}}\right )$. 


Thus,

$$ \frac{\partial}{\partial w_{n}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )y_{p}$$

$$ =  - \begin{bmatrix} 1 &1  &...  &1 \end{bmatrix}_{1 \times P}$$

$$ \cdot \begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot tanh\left ( c_{n}+\mathbf{x_{1}^{T}}\mathbf{v_{n}} \right )\cdot y_{1} &...  & \sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot tanh\left ( c_{n}+\mathbf{x_{P}^{T}}\mathbf{v_{n}} \right ) \cdot y_{P}
\end{bmatrix}^{T}$$

$$ =-\mathbf{1_{P\times 1}^{T}}\left ( \mathbf{q}\odot \mathbf{t_{n}}\odot\mathbf{y} \right )$$

where $tanh\left ( c_{m}+\mathbf{x_{p}^{T}v_{m}} \right )=a\left (  c_{m}+\mathbf{x_{p}^{T}v_{m}}\right )$ and $tanh\left ( c_{n}+\mathbf{x_{p}^{T}v_{n}} \right )=a\left (  c_{n}+\mathbf{x_{p}^{T}v_{n}}\right )$.  

Thus,

$$\frac{\partial}{\partial c_{n}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )w_{n}y_{p}$$

$$ =  - \begin{bmatrix} 1 &1  &...  &1 \end{bmatrix}_{1 \times P}$$

$$ \cdot \begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot sech^{2}\left ( c_{n}+\mathbf{x_{1}^{T}}\mathbf{v_{n}} \right )\cdot w_{n}\cdot y_{1} &...  & \sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot sech^{2}\left ( c_{n}+\mathbf{x_{P}^{T}}\mathbf{v_{n}} \right ) \cdot w_{n}\cdot y_{P}
\end{bmatrix}^{T}$$

$$ =-\mathbf{1_{P\times 1}^{T}}\left ( \mathbf{q}\odot \mathbf{s_{n}}\odot\mathbf{y} \right )\cdot w_{n}$$

where $sech^{2}\left ( c_{n}+\mathbf{x_{p}^{T}v_{n}} \right )=a'\left (  c_{n}+\mathbf{x_{p}^{T}v_{n}}\right )$ and $tanh\left ( c_{m}+\mathbf{x_{p}^{T}v_{m}} \right )=a\left (  c_{m}+\mathbf{x_{p}^{T}v_{m}}\right )$. 

Thus,

$$\nabla_{\mathbf{v_{n}}}g=-\sum_{p=1}^{P}\sigma \left ( -y_{p} \left ( b+\sum_{m=1}^{M}w_{m}a\left ( c_{m}+\mathbf{x_{p}^{T}}\mathbf{v_{m}}\right ) \right )\right )a'\left ( c_{n}+\mathbf{x_{p}^{T}}\mathbf{v_{n}}\right )\mathbf{x_{p}}w_{n}y_{p}$$ 

$$ =  - \begin{bmatrix} x_{1} &x_{2}  &...  &x_{P} \end{bmatrix}_{2 \times P}$$

$$ \cdot \begin{bmatrix}
\sigma \left ( -y_{1}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{1}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot sech^{2}\left ( c_{n}+\mathbf{x_{1}^{T}}\mathbf{v_{n}} \right )\cdot w_{n}\cdot y_{1} &...  & \sigma \left ( -y_{P}\left ( b+\sum_{m=1}^{M}w_{m}tanh\left ( c_{m}+\mathbf{x_{P}^{T}}\mathbf{v_{m}} \right ) \right ) \right )\cdot sech^{2}\left ( c_{n}+\mathbf{x_{P}^{T}}\mathbf{v_{n}} \right ) \cdot w_{n}\cdot y_{P}
\end{bmatrix}^{T}$$

$$ =-\mathbf{X}\cdot \mathbf{q}\odot \mathbf{s_{n}}\odot\mathbf{y}\cdot w_{n}$$

End.

Based on the derivatives above, the corresponding code (using gradient descent) in Python is as follows,

```Python
	b = 0
	w = np.random.rand(M, 1)
	c = np.zeros((M, 1))
	V = np.random.rand(M, 2)
	P = np.size(y)
	alpha = 1e-2
	l_P = np.ones((P, 1))
	max_its = 10000
	k = 1
    
	for k in range(max_its):
		q = np.zeros((P,1))
		for p in np.arange(0,P):
			x = X[p].reshape(1,np.size(X[p]))
			q[p] = sigmoid(-y[p] * (b + np.dot(w.T, np.tanh(c + np.dot(V, x.T)))))
		grad_b = -1 * np.dot(l_P.T, q * y)
		grad_w = np.zeros((M, 1))
		grad_c = np.zeros((M, 1))
		grad_V = np.zeros((M, 2))
		for n in np.arange(0, M):
			v = V[n].reshape(2,1)
			t = np.tanh(c[n] +np.dot(X,v))
			s = 1 / np.cosh(c[n]+np.dot(X,v))**2
			grad_w[n] = -1 * np.dot(l_P.T,q * t * y)
			grad_c[n] = -1 * np.dot(l_P.T,q * s * y) * w[n]
			grad_V[n] = (-1 * np.dot(X.T, q * s * y) * w[n]).reshape(2,)          
		b = b - alpha * grad_b
		w = w - alpha * grad_w
		c = c - alpha * grad_c
		V = V - alpha * grad_V
		k = k + 1
  ```

## k-nearest neighbors (k-NN)

The k-nearest neighbors (k-NN) is a local classification scheme, while differing from the more global feature basis approach, can produce non-linear boundaries in the original feature space.

With the k-NN approach there is no training phase to the classification scheme. We simply use the training data directly to classify any new point $x_{new}$ by taking the average of the labels of its k-nearest neighbors. That is, we create the label $y_{new}$ for a point $x_{new}$ by simply calculating

$$ y_{new}=sign\left ( \sum _{i\epsilon \Omega } y_{i}\right ),$$

where \Omega is the set of indices of the k closest traning points to $x_{new}. To avoid tie votes (i.e., a value of zero above) typically the number of neighbors k is chosen to be odd. The corresponding code for k-NN in Python is as follows,

```Python
def knn(data, x, y, k):
    sum = 0
    m = np.zeros((len(data), 2))
    for i in np.arange(len(data)):
        m[i][0] = (x - data[i][0])**2 + (y - data[i][1])**2
        m[i][1] = data[i][2]
    m = m.tolist()
    m.sort(key=lambda x: x[0])
    m = np.asarray(m)
    for i in range(0, k):
        sum=sum+ m[i][1]
    ynew = sign(sum)
    return ynew
  ```
