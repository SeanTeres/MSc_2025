import networks
import torch
import torch.nn as nn


class DRIT(nn.Module):
    def __init__(self, cfg):
        super(DRIT, self).__init__()

        self.cfg = cfg
        
        # DEFAULT PARAMS FROM SRC
        
        lr = 0.0001
        lr_D_content = lr / 2.5
        self.nz = 8
        self.concat = False
        self.no_ms = True

        ############################
        # DISCRIMINATORS
        ############################

        if cfg["DISC_SCALE"] > 1:
        # USE MULTI-SCALE DISCRIMINATOR(S)
            self.disc_A = networks.MultiScaleDis(cfg["INPUT_DIM_A"], cfg["DISC_SCALE"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
            self.disc_B = networks.MultiScaleDis(cfg["INPUT_DIM_B"], cfg["DISC_SCALE"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])   
            self.disc_A2 = networks.MultiScaleDis(cfg["INPUT_DIM_A"], cfg["DISC_SCALE"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
            self.disc_B2 = networks.MultiScaleDis(cfg["INPUT_DIM_B"], cfg["DISC_SCALE"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])   
        else:
            self.disc_A = networks.Dis(cfg["INPUT_DIM_A"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
            self.disc_B = networks.Dis(cfg["INPUT_DIM_B"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
            self.disc_A2 = networks.Dis(cfg["INPUT_DIM_A"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
            self.disc_B2 = networks.Dis(cfg["INPUT_DIM_B"], norm=self.cfg["DISC_NORM"], sn=self.cfg["SPECTRAL_NORM"])
        
        self.disc_content = networks.Dis_content()

        ############################
        # ENCODERS
        ############################

        self.enc_content = networks.E_content(self.cfg["INPUT_DIM_A"], self.cfg["INPUT_DIM_B"])

        if self.cfg["VAE_BASED"]:
           self.enc_attr = networks.E_attr_concat(self.cfg["INPUT_DIM_A"], self.cfg["INPUT_DIM_B"], self.nz, norm_layer=None,
                                                    nl_layer=networks.get_non_linearity(layer_type='lrelu'))  # Fixed: was n1_layer
        else:
           self.enc_attr = networks.E_attr(self.cfg["INPUT_DIM_A"], self.cfg["INPUT_DIM_B"], self.nz)

        ############################
        # GENERATORS
        ############################

        if self.concat:
          self.gen = networks.G_concat(self.cfg["INPUT_DIM_A"], self.cfg["INPUT_DIM_B"], nz=self.nz)
        else:
          self.gen = networks.G(self.cfg["INPUT_DIM_A"], self.cfg["INPUT_DIM_B"], nz=self.nz)

        ############################
        # OPTIMIZERS
        ############################

        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disc_A2_opt = torch.optim.Adam(self.disc_A2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disc_B2_opt = torch.optim.Adam(self.disc_B2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disc_content_opt = torch.optim.Adam(self.disc_content.parameters(), lr=lr_D_content, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_content_opt = torch.optim.Adam(self.enc_content.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_attr_opt = torch.optim.Adam(self.enc_attr.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)  

        ############################
        # SETUP LOSS  # Fixed typo: was SEETUP
        ############################

        self.criterionL1 = torch.nn.L1Loss()



        
    ############################
    # FUNCTIONS
    ############################

    def initialize_weights(self):
        self.disc_A.apply(networks.gaussian_weights_init)
        self.disc_B.apply(networks.gaussian_weights_init)
        self.disc_A2.apply(networks.gaussian_weights_init)
        self.disc_B2.apply(networks.gaussian_weights_init)
        self.disc_content.apply(networks.gaussian_weights_init)
        self.enc_content.apply(networks.gaussian_weights_init)
        self.enc_attr.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        print("Initialized model weights")

    def set_scheduler(self, cfg, last_epoch=-1):
        self.disc_A_scheduler = networks.get_scheduler(self.disc_A_opt, cfg, cur_ep=last_epoch)
        self.disc_B_scheduler = networks.get_scheduler(self.disc_B_opt, cfg, cur_ep=last_epoch)
        self.disc_A2_scheduler = networks.get_scheduler(self.disc_A2_opt, cfg, cur_ep=last_epoch)
        self.disc_B2_scheduler = networks.get_scheduler(self.disc_B2_opt, cfg, cur_ep=last_epoch)
        self.disc_content_scheduler = networks.get_scheduler(self.disc_content_opt, cfg, cur_ep=last_epoch)
        self.enc_content_scheduler = networks.get_scheduler(self.enc_content_opt, cfg, cur_ep=last_epoch)
        self.enc_attr_scheduler = networks.get_scheduler(self.enc_attr_opt, cfg, cur_ep=last_epoch)
        self.gen_scheduler = networks.get_scheduler(self.gen_opt, cfg, cur_ep=last_epoch)
        print("Set up learning rate schedulers")

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        z = torch.randn(batch_size, nz).to(self.device)
        return z

    def test_forward(self, image, A_2_B=True):
        self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss').to(self.device)

        if A_2_B:
            # Translate from A to B 

            # Get content representation from domain A
            self.z_content = self.enc_content.forward_a(image)
            # Generate fake image in domain B using content from A and random attribute 
            output = self.gen.forward_b(self.z_content, self.z_random)

        else:
            # Translate from B to A
            
            # Get content representation from domain B
            self.z_content = self.enc_content.forward_b(image)
            # Generate fake image in domain A using content from B and random attribute 
            output = self.gen.forward_a(self.z_content, self.z_random)
        
        return output


    def forward(self):
        # input_images
        """
        input_A: BATCH of real images from domain A
        input_B: BATCH of real images from domain B

        half_size: First half are used as encoded samples, second half used as random samples for pairing

        """

        half_size = 1
        real_A, real_B = self.input_A, self.input_B

        self.real_A_encoded = real_A[:half_size]
        self.real_B_encoded = real_B[:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_random = real_B[half_size:]

        self.z_content_A, self.z_content_B= self.enc_content.forward(self.real_A_encoded, self.real_B_encoded)

        if self.concat:
        # VAE BASED
            raise NotImplementedError("Concat not implemented")
        else:
            self.z_attr_A, self.z_attr_B = self.enc_attr.forward(self.real_A_encoded, self.real_B_encoded)
        

        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss').to(self.device)

        if not self.no_ms:
            self.z_random_2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss').to(self.device)

        if not self.no_ms:
            input_content_for_A = torch.cat((self.z_content_B, self.z_content_A, self.z_content_B, self.z_content_B), 0)
            input_content_for_B = torch.cat((self.z_content_A, self.z_content_B, self.z_content_A, self.z_content_A), 0)

            input_attr_for_A = torch.cat((self.z_attr_A, self.z_attr_A, self.z_random, self.z_random_2), 0)
            input_attr_for_B = torch.cat((self.z_attr_B, self.z_attr_B, self.z_random, self.z_random_2), 0)

            output_fake_A = self.gen.forward_a(input_content_for_A, input_attr_for_A)
            output_fake_B = self.gen.forward_b(input_content_for_B, input_attr_for_B)

            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_A_random_2 = torch.split(output_fake_A, self.z_content_A.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_B_random_2 = torch.split(output_fake_B, self.z_content_B.size(0), dim=0)
        else:
            
            input_content_for_A = torch.cat((self.z_content_B, self.z_content_A, self.z_content_B), 0)
            input_content_for_B = torch.cat((self.z_content_A, self.z_content_B, self.z_content_A), 0)

            input_attr_for_A = torch.cat((self.z_attr_A, self.z_attr_A, self.z_random), 0)
            input_attr_for_B = torch.cat((self.z_attr_B, self.z_attr_B, self.z_random), 0)

            output_fake_A = self.gen.forward_a(input_content_for_A, input_attr_for_A)
            output_fake_B = self.gen.forward_b(input_content_for_B, input_attr_for_B)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fake_A, self.z_content_A.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fake_B, self.z_content_B.size(0), dim=0)

        
        # GET THE RECONSTRUCTED ENCODED Z_CONTENT
        self.z_content_recon_B, self.z_content_recon_A = self.enc_content.forward(self.fake_A_encoded, self.fake_B_encoded)

        # GET THE RECONSTRUCTED ENCODED Z_ATTR
        if self.concat:
            raise NotImplementedError("Concat not implemented")
        else:
            self.z_attr_recon_A, self.z_attr_recon_B = self.enc_attr.forward(self.fake_A_encoded, self.fake_B_encoded)

        
                
        ############################
        # 2ND CROSS-TRANSLATION FOR CYCLE CONSISTENCY
        ############################

        self.fake_A_recon = self.gen.forward_a(self.z_content_recon_A, self.z_attr_recon_A)
        self.fake_B_recon = self.gen.forward_b(self.z_content_recon_B, self.z_attr_recon_B)

                
        ############################
        # FOR IMAGE DISPLAY
        ############################
        self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                        self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                        self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                        self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)
        
        # LATENT REGRESSION GOES HERE
        if self.concat:
            raise NotImplementedError("Concat not implemented")
        else:
            self.z_attr_random_A, self.z_attr_random_B = self.enc_attr.forward(self.fake_A_random, self.fake_B_random)

    def forward_content(self):
        half_size = 1
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]

        # GET ENCODED Z_CONTENT
        self.z_content_A, self.z_content_B = self.enc_content.forward(self.real_A_encoded, self.real_B_encoded)
        
    def update_D_content(self, image_A, image_B):
        self.input_A = image_A
        self.input_B = image_B

        self.forward_content()
        self.disc_content_opt.zero_grad()
        loss_D_content = self.backward_D_content(self.z_content_A, self.z_content_B)
        
        # GRADIENT CLIPPING
        nn.utils.clip_grad_norm_(self.disc_content.parameters(), 5)
        self.disc_content_opt.step()

    def update_D(self, image_A, image_B):
        self.input_A = image_A
        self.input_B = image_B

        self.forward()

        # UPDATE DISC_ATTR
        self.disc_A_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disc_A, self.real_A_encoded, self.fake_A_encoded)
        self.dis_A_loss = loss_D1_A.item()
        self.disc_A_opt.step()

        # UPDATE DISC_ATTR2
        self.disc_A2_opt.zero_grad()
        loss_D2_A = self.backward_D(self.disc_A2, self.real_A_random, self.fake_A_random)
        self.dis_A2_loss = loss_D2_A.item()
        
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(self.disc_A2, self.real_A_random, self.fake_A_random_2)
            self.dis_A2_loss += loss_D2_A2.item()
        self.disc_A2_opt.step()

        # UPDATE DISC_B
        self.disc_B_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disc_B, self.real_B_encoded, self.fake_B_encoded)
        self.dis_B_loss = loss_D1_B.item()
        self.disc_B_opt.step()

        # UPDATE DISC_B2
        self.disc_B2_opt.zero_grad()
        loss_D2_B = self.backward_D(self.disc_B2, self.real_B_random, self.fake_B_random)
        self.dis_B2_loss = loss_D2_B.item()

        if not self.no_ms:
            loss_D2_B2 = self.backward_D(self.disc_B2, self.real_B_random, self.fake_B_random_2)
            self.dis_B2_loss += loss_D2_B2.item()
        self.disc_B2_opt.step()

        # Update DISC_CONTENT
        self.disc_content_opt.zero_grad()
        loss_D_content = self.backward_D_content(self.z_content_A, self.z_content_B)
        self.disc_content_loss = loss_D_content.item()
        self.disc_content_opt.step()


    def backward_D(self, netD, real, fake):

        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)

        loss_D = 0
        for item, (out_A, out_B) in enumerate(zip(pred_fake, pred_real)):
            # Remove sigmoid and use BCE with logits instead
            all0 = torch.zeros_like(out_A).to(self.device)
            all1 = torch.ones_like(out_B).to(self.device)

            # Use binary_cross_entropy_with_logits which combines sigmoid + BCE
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_A, all0)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_B, all1)
            loss_D += ad_true_loss + ad_fake_loss
        # loss_D.backward()
        return loss_D

    def backward_D_content(self, imageA, imageB):
        pred_fake = self.disc_content.forward(imageA.detach())
        pred_real = self.disc_content.forward(imageB.detach())

        for item, (out_A, out_B) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_A)
            out_real = nn.functional.sigmoid(out_B)

            all0 = torch.zeros_like(out_A).to(self.device)
            all1 = torch.ones_like(out_B).to(self.device)

            # Use binary_cross_entropy_with_logits which combines sigmoid + BCE
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_A, all0)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_B, all1)

        loss_D_content = ad_true_loss + ad_fake_loss
        # loss_D_content.backward()
        return loss_D_content

    def update_enc_gen(self):
        # UPDATE EC, EA and GEN
        self.enc_content_opt.zero_grad()
        self.enc_attr_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_enc_gen()
        self.enc_content_opt.step()
        self.enc_attr_opt.step()
        self.gen_opt.step()

        # UPDATE G, EC
        self.enc_content_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_G_alone()
        self.enc_content_opt.step()
        self.gen_opt.step()


    def backward_enc_gen(self):
        # CONTENT ADV LOSS FOR GENERATOR

        loss_G_GAN_A_content = self.backward_G_GAN_content(self.z_content_A)
        loss_G_GAN_B_content = self.backward_G_GAN_content(self.z_content_B)

        # LOSS ADV FOR GENERATOR
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disc_A)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disc_B)
        

        # KL LOSS: Z_A
        if self.concat:
            raise NotImplementedError("Concat not implemented")
        else:
            loss_kl_za_A = self._l2_regularize(self.z_attr_A) * 0.01
            loss_kl_za_B = self._l2_regularize(self.z_attr_B) * 0.01

        # KL LOSS: Z_C
        loss_kl_zc_A = self._l2_regularize(self.z_content_A) * 0.01
        loss_kl_zc_B = self._l2_regularize(self.z_content_B) * 0.01


        # CROSS-CYCLE CONSISTENCY LOSS
        loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
        loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10

        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + loss_G_GAN_A_content + loss_G_GAN_B_content + \
                    loss_G_L1_AA + loss_G_L1_BB + loss_G_L1_A + loss_G_L1_B + \
                    loss_kl_za_A + loss_kl_za_B + loss_kl_zc_A + loss_kl_zc_B
        
        # loss_G.backward(retain_graph=True)

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_A_content = loss_G_GAN_A_content.item()
        self.gan_loss_B_content = loss_G_GAN_B_content.item()
        self.kl_loss_za_A = loss_kl_za_A.item()
        self.kl_loss_za_B = loss_kl_za_B.item()
        self.kl_loss_zc_A = loss_kl_zc_A.item()
        self.kl_loss_zc_B = loss_kl_zc_B.item()

        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()

        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        outs = self.disc_content.forward(data)

        for output in outs:
            all_half = 0.5 * torch.ones_like(output).to(self.device)
            ad_loss = nn.functional.binary_cross_entropy_with_logits(output, all_half)
                
        return ad_loss


    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0

        for out_a in outs_fake:
                # Remove sigmoid and use BCE with logits
                all1 = torch.ones_like(out_a).to(self.device)
                loss_G += nn.functional.binary_cross_entropy_with_logits(out_a, all1)
        return loss_G


    def backward_G_alone(self):

        # ADVERSARIAL LOSS FOR GENERATOR

        loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disc_A2)
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disc_B2)

        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random_2, self.disc_A2)
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random_2, self.disc_B2)
            loss_G = loss_G_GAN2_A + loss_G_GAN2_B + loss_G_GAN2_A2 + loss_G_GAN2_B2

        if not self.no_ms:
            # MODE SEELING LOSS FOR A2B AND B2A
            lz_AB = torch.mean(torch.abs(self.fake_B_random_2 - self.fake_B_random) / torch.mean(torch.abs(self.z_random_2 - self.z_random)))
            lz_BA = torch.mean(torch.abs(self.fake_A_random_2 - self.fake_A_random) / torch.mean(torch.abs(self.z_random_2 - self.z_random)))
            
            eps = 1*1e-5

            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)

        # LATENT REGRESSION LOSS
        if self.concat:
            raise NotImplementedError("Concat not implemented")
        else:
            loss_z_L1_A = torch.mean(torch.abs(self.z_attr_random_A - self.z_attr_A)) * 10
            loss_z_L1_B = torch.mean(torch.abs(self.z_attr_random_B - self.z_attr_B)) * 10

        loss_z_L1 = loss_z_L1_A + loss_z_L1_B + loss_G_GAN2_A + loss_G_GAN2_B

        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        # loss_z_L1.backward()

        self.l1_recon_z_loss_a = loss_z_L1_A.item()
        self.l1_recon_z_loss_b = loss_z_L1_B.item()

        if not self.no_ms:
            self.gan2_loss_a = (loss_G_GAN2_A).item()
            self.gan2_loss_b = (loss_G_GAN2_B).item()

    def update_lr(self):
        self.disc_A_scheduler.step()
        self.disc_B_scheduler.step()
        self.disc_A2_scheduler.step()
        self.disc_B2_scheduler.step()
        self.disc_content_scheduler.step()
        self.enc_content_scheduler.step()
        self.enc_attr_scheduler.step()
        self.gen_scheduler.step()
        
    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def save(self, filename, ep, total_it):
        state = {
            'disc_A': self.disc_A.state_dict(),
            'disc_B': self.disc_B.state_dict(),
            'disc_A2': self.disc_A2.state_dict(),
            'disc_B2': self.disc_B2.state_dict(),
            'disc_content': self.disc_content.state_dict(),
            'enc_content': self.enc_content.state_dict(),
            'enc_attr': self.enc_attr.state_dict(),
            'gen': self.gen.state_dict(),
            'disc_A_opt': self.disc_A_opt.state_dict(),
            'disc_B_opt': self.disc_B_opt.state_dict(),
            'disc_A2_opt': self.disc_A2_opt.state_dict(),
            'disc_B2_opt': self.disc_B2_opt.state_dict(),
            'disc_content_opt': self.disc_content_opt.state_dict(),
            'enc_content_opt': self.enc_content_opt.state_dict(),
            'enc_attr_opt': self.enc_attr_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        print(f"Saved model checkpoint at epoch {ep}, total_it {total_it} to {filename}")
        return
