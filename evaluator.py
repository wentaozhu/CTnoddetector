import os
import pandas as pd
import numpy as np
import sys
sys.path.append('../tcai17/')
import utils
import matplotlib.pyplot as plt
import time

def is_hit(c_hat, c, d):
    """
    Hit criterion for LUNA16.

    c_hat: predicted center x,y,z
    c:     ground truth center
    d:     diameter of ground truth nodule
    """
    return np.linalg.norm(c_hat - c) < d/2.0

def s_to_p(s):
    """Convert scores array to probabilities."""
    from torch.autograd import Variable
    from torch import from_numpy
    from torch.nn import Sigmoid
    m = Sigmoid()
    input = Variable(from_numpy(s))
    return  m(input).data.numpy()

class Evaluator:
    """Suite of analytics for completed predictions.

    E.g., generate FROC statistics, get false positives, etc.
    """

    BASE = '/home/danielrk/tc/nod/training/detector/'

    # dir of preprocessed imgs to load spacing, ebox_origin, origin
    #PREP = '/home/danielrk/lung/input/tianchi_prep_HU_pres/'

    # dir of .mhd raw files
    RAW = '/home/danielrk/lung/input/tianchi_combined/'

    def __init__(self, results_name, test_set, prep_path, ckpt=None, pbb_cutoff=None, topk = None, classifier_pred_path = None, bbox_dir = None, apply_sigmoid=True):
        """
        results_name: dir with saved predictions, eg '05-31-17'
        test_set: 'test', 'val', 'train'
        ckpt: int for name of bbox dir, eg 061817/bbox/test2_100; none if just 'test2'
        pbb_cutoff: only load pbbs with prob > cutoff for efficiency
        """
        self.PREP = prep_path
        print 'USING',self.PREP,'FOR SPACING, ORIGIN, EBOX_ORIGIN'
        self.test_set = test_set
        self.results_dir = os.path.join(self.BASE, 'results', results_name)
        self.bbox_dir = os.path.join(self.BASE, 'results',
                                     results_name, 'bbox/') \
                                             if bbox_dir is None else bbox_dir
        self.pbb_cutoff = pbb_cutoff
        self.topk = topk
        self.classifier_pred_path = classifier_pred_path
        self.apply_sigmoid = apply_sigmoid

        sd = os.path.join(self.BASE, 'results', results_name, 'sub')
        if not os.path.exists(sd):
            os.makedirs(sd)

        suf = '' if ckpt is None else '_%03d' % ckpt
        if bbox_dir is None:
            self.bbox_dir = os.path.join(self.bbox_dir, test_set + suf)
        self.filenames = np.load(os.path.join(self.BASE,'filenames_'+test_set+'.npy'))
        self.save_dir = os.path.join(sd, suf + '_' + str(int(time.time())) + '.csv')
        print 'save_dir for csv:', self.save_dir
        #if test_set == 'train' or test_set == 'val':
        self.generate_stats()

    def generate_stats(self):
        """Store PBBs in dataframe (self.pbbs) along with ground truth binary labels.

        self.n_annot : total number of nodule annotations

        self.rel = pbbs_df[pbbs_df['nod']==1]
        self.irr = pbbs_df[pbbs_df['nod']==0]
        self.lbbs_not_in_pbbs = lbbs_not_in_pbbs_df

        classifier_pred_path: optionally include classifier prediction scores
        """

        #if self.test_set == 'test':
        #    print 'Error: test set has no labels'
        #    return

        lbbs_not_in_pbbs_df = pd.DataFrame(columns=['pid','z','y','x','d'])
        if self.classifier_pred_path is None:
            pbbs_df = pd.DataFrame(columns=['pid','prob','z','y','x','d','nod'])
        else:
            print 'WARNING: ASSUMES CLASSIFIER PRED MATCHES WITH SORTED PBB'
            pbbs_df = pd.DataFrame(columns=['pid','prob','z','y','x','d','nod','c_prob'])

        n_annot = 0
        for name in self.filenames:
            #print name
            pbb = np.load(os.path.join(self.bbox_dir, name+'_pbb.npy'))
            # add nod
            pbb = np.concatenate([pbb, np.zeros((pbb.shape[0],1))], axis=1)
            pbb = pbb[pbb[:,0].argsort()][::-1]


            # Include classifier scores
            # Use nan for patients that got pbbs but not classifier predictions
            # eg blacklist

            if self.classifier_pred_path is not None:
                pred_fname = os.path.join(self.classifier_pred_path, name+'_pred.npy')
                if os.path.exists(pred_fname):
                    cl_scores = np.load(pred_fname)
                else:
                    cl_scores = np.empty((pbb.shape[0],1))
                    cl_scores[:] = np.nan

                #print 'pbb:{}, cl_scores:{}'.format(pbb.shape, cl_scores.shape)
                if self.topk is not None:
                    pbb = np.concatenate([pbb[:self.topk],cl_scores[:self.topk]], axis=1)
                else:
                    print 'error: if CPP supplied, then topk must be used to match cl_scores'
                    return
            lbb = np.load(os.path.join(self.bbox_dir, name+'_lbb.npy'))
            n_annot += len(lbb)
            lab_hits = np.zeros(len(lbb))

            # determine ground truth label of pbb
            # exclude relevant pbbs that are redundant for purposes of FROC

            #print 'pbb len', len(pbb)
            it = range(len(pbb)) if self.topk is None else range(min(len(pbb),self.topk))
            for i in it:

                if self.pbb_cutoff is not None and pbb[i,0] < self.pbb_cutoff:
                    break

                lbb_match = False
                redundant_hit = False
                for j in range(len(lbb)):
                    if is_hit(pbb[i][1:4], lbb[j][:3], lbb[j][3]):
                        if lab_hits[j] > 0:
                            redundant_hit = True
                            #print 'redundant tp!'
                            #print name, 'pbb', pbb[i], 'lbb', lbb[j]
                            #tp.append(pbb[i])
                        lab_hits[j] += 1
                        lbb_match = True
                        break
                if lbb_match:
                    pbb[i,5] = 1
                else:
                    pbb[i,5] = 0

                if not redundant_hit:
                    pbbs_df.loc[len(pbbs_df)] = [name] + list(pbb[i])
            missed = pd.DataFrame(columns=list('zyxd'), data = lbb[lab_hits == 0].reshape(-1,len(list('zyxd'))))
            missed['pid'] = name
            missed = missed[['pid','z','y','x','d']]
            lbbs_not_in_pbbs_df = pd.concat([lbbs_not_in_pbbs_df,missed], ignore_index=True)


        # convert scores to probabilities
        if self.apply_sigmoid:
            pbbs_probs = s_to_p(np.array(pbbs_df['prob']))
            pbbs_df['prob'] = pbbs_probs

        if self.classifier_pred_path is not None:
            pbbs_cprobs = s_to_p(np.array(pbbs_df['c_prob']))
            pbbs_df['c_prob'] = pbbs_cprobs

            # ensemble
            pbbs_df['ensemble'] = (pbbs_df['prob'] + pbbs_df['c_prob'])/2.0
            pbbs_df['det8'] = (pbbs_df['prob']**8 + pbbs_df['c_prob'])/2.0



        self.n_annot = n_annot
        self.pbbs = pbbs_df
        self.rel = pbbs_df[pbbs_df['nod']==1]
        self.irr = pbbs_df[pbbs_df['nod']==0]
        self.lbbs_not_in_pbbs = lbbs_not_in_pbbs_df
        print 'loaded {} pbbs'.format(len(pbbs_df))
        if self.test_set == 'train' or self.test_set == 'val':
            print 'saved pbbs missed {} out of {} annotations ({:.2%})'.format(len(lbbs_not_in_pbbs_df),
                                                                       n_annot,
                                                                           1.0 * len(lbbs_not_in_pbbs_df)/n_annot)




    def froc(self, by='prob', ignore=[], n_scans=None):
        """Print FROC statistics and return (fp_per_scan,
                                             TPRs,
                                             probability_thresholds).
        by: 'prob' , 'c_prob', 'ensemble'
        ignore: list of patients to ignore pbbs for FROC
        n_scans: # scans used for fp_per_scan. If None, use len(self.filenames)
        """

        if self.test_set == 'test':
            print 'Error: test set has no labels'
            return

        irr = self.irr.loc[~self.irr['pid'].isin(ignore)]
        rel = self.rel.loc[~self.rel['pid'].isin(ignore)]

        irr = np.array(irr[by])
        rel = np.array(rel[by])
        irr = irr[irr.argsort()][::-1]
        rel = rel[rel.argsort()][::-1]
        tprs = []
        p_ths = []
        fp_per_scan = [1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0, 8.0]
        if n_scans is None:
            n_scans = len(self.filenames)
        for nlf in fp_per_scan:
            irr_i = int(np.round(nlf * n_scans))
            # if not enough false positives, assume padded false positive list
            # with p=0
            prob_th = 0 if irr_i >= len(irr) else irr[irr_i]
            tpr = np.sum(rel > prob_th)/(1.0 * self.n_annot)
            tprs.append(tpr)
            p_ths.append(prob_th)
            print 'NLF: {}, TPR: {}, PROB_TH: {}'.format(nlf, tpr, prob_th)
        print '======'
        print 'avg TPR: {}'.format(np.mean(tprs))

        return (fp_per_scan, tprs, p_ths)

        #plt.plot(fp_per_scan, tprs)
        #plt.show()

