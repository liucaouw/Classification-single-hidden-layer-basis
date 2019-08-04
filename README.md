# Classification using single hidden layer basis

When employng M single hidden layer basis features (using any activation a(.)) the full gradient of a cost g (e.g., the softmax) is a vector of length Q=M(N+2)+1 containing the derivation of the cost with respect to each variable.

$$\nabla g=\begin{bmatrix}
\frac{\partial g}{\partial b} & \frac{\partial g}{\partial w_{1}} & ... & \frac{\partial g}{\partial w_{M}} & \frac{\partial g}{\partial c_{1}} &  ... & \frac{\partial g}{\partial c_{M}} & \frac{\nabla g^{T}}{\nabla \mathbf{v_{1}} } & ...  & \frac{\nabla g^{T}}{\nabla \mathbf{v_{M}} }
\end{bmatrix}^{T}$$
