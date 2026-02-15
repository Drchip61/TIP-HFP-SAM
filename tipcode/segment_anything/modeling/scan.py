def forward_single(self,x:torch.Tensor):
    selective_scan = selective_scan_fn
    B, L, d = x.shape
    x = x.permute(0, 2, 1)
    x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
dt = self.dt_proj.weight @ dt.t()
dt = rearrange(dt, "d (b l) -> b d l", l=L)
A = -torch.exp(self.A_log.float())  # (k * d, d_state)
B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

y = selective_scan(
    x, dt,
    A, B, C, self.D.float(),
    delta_bias=self.dt_proj.bias.float(),
    delta_softplus=True,
)
# assert out_y.dtype == torch.float
y = rearrange(y, "b d l -> b l d")
y = self.out_norm(y)
return y
