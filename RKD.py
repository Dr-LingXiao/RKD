
criterion1 = nn.MSELoss()
criterion2 = torch.nn.MarginRankingLoss(margin = 0.2, reduction="none")

def RFD(embed_x, embed_y, embed_z, embed_x_T, embed_y_T, embed_z_T, criterion1, criterion2): # x: anchor y: far, z: close
  
  sim_a = torch.sum(embed_x * embed_y, dim=1)     
  sim_b = torch.sum(embed_x * embed_z, dim=1)
   
  loss_triplet_S = criterion2(sim_a, sim_b, target)

  sim_a_T = torch.sum(x_T * y_T, dim=1)
  sim_b_T = torch.sum(x_T * z_T, dim=1)


            loss_triplet_T = self.criterion2(sim_a_T, sim_b_T, target)

            score1 = (self.criterion1(loss_triplet_S, loss_triplet_T.detach()) + self.criterion1(loss_triplet_T, loss_triplet_S.detach()))/2.

            score2 = (self.criterion1(sim_a, sim_a_T.detach()) + self.criterion1(sim_b, sim_b_T.detach()) + self.criterion1(sim_a_T, sim_a.detach()) + self.criterion1(sim_b_T, sim_b.detach()))/4.
