"""Build a twelve-page report with computed results and explicit source mapping."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import matplotlib
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4


def build_report(cfg,output):
    output=Path(output);student=cfg['student'];sim=pd.read_csv(output/'simulation_raw.csv')
    checkpoints=pd.read_csv(output/'dynamics_checkpoints.csv')
    decisions=json.loads((output/'model_decisions.json').read_text())
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Text','DejaVuSans.ttf'),('Strong','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Text',normal='Text',bold='Strong',italic='Italic',boldItalic='Strong')
    body=ParagraphStyle('body',fontName='Text',fontSize=9.5,leading=14.2,spaceAfter=9,textColor=colors.HexColor('#172B3A'))
    heading=ParagraphStyle('heading',parent=body,fontName='Strong',fontSize=18,leading=23,spaceAfter=15)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11,spaceAfter=7)
    sub=ParagraphStyle('sub',parent=body,fontName='Strong',fontSize=11,leading=16,spaceBefore=5)
    story=[]; markdown=[]
    def text(s,style=body):
        story.append(Paragraph(s,style));markdown.append(s)
    def title(s):
        if story:story.append(PageBreak())
        text(s,heading);markdown.append('\n')
    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(x)),small) for x in row] for row in [headers]+rows]
        tab=Table(data,colWidths=widths or [495/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EFF4')),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#205375')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#9AAAB6'))]))
        story.extend([tab,Spacer(1,10)]);markdown.append(pd.DataFrame(rows,columns=headers).to_csv(index=False))
    def figure(name,caption,width=495):
        from PIL import Image as PILImage
        file=output/'figures'/f'{name}.png'
        with PILImage.open(file) as im:height=width*im.height/im.width
        story.append(Image(str(file),width=width,height=height));text(caption,small)
        markdown.append(f'Figure: figures/{name}.pdf')
    def mean(strength,model,penalty,cq,metric):
        s=sim[(sim.strength==strength)&(sim.model==model)&(sim.cq==cq)]
        if penalty is not None:s=s[np.isclose(s.penalty,penalty)]
        return s[metric].mean()
    def f(x):return f'{x:.4f}'
    title('Complexity, Shrinkage and Learning Time\n in Financial Return Prediction')
    text('DSA5205 · Project 1 · September 2026',sub)
    text(f'<b>{student["name"]}</b><br/>Student ID: {student["id"]}<br/>{student["email"]}')
    story.append(Spacer(1,16))
    text('Abstract',sub)
    text('This project studies when greater model complexity improves return prediction and when it instead amplifies estimation noise. Paired finite-sample simulations reproduce the interpolation instability of ridgeless regression and the improvement available when additional signals are combined with appropriate ridge shrinkage. Lasso provides a contrasting sparsity bias on the same dense-signal data. A separate training-only validation procedure selects models for the three supplied prediction tasks. A finite two-layer-network experiment then relates learning time to overfitting and to changes in the latent return signal.')
    text(f'With signal strength b*=1, the mean ridgeless out-of-sample R² is {mean(1,"ridge",0,1.02,"r2"):.2f} just above interpolation. At observed complexity 10, ridge with z=10 obtains R²={f(mean(1,"ridge",10,10,"r2"))} and paper Sharpe={f(mean(1,"ridge",10,10,"sharpe"))}. Pair A has positive validation evidence; the evidence for B and C is weak. Neural early stopping improves stationary test error, but does not eliminate the loss from a rotated signal direction. All reported empirical quantities come from executed experiments; hidden-test labels were unavailable.')
    text('Central argument',sub)
    text('Parameter count alone does not determine financial usefulness. Added inputs can reduce omitted-signal error, while shrinkage controls variance. For a nonlinear learner, training time also changes the fitted representation. These benefits rely on assumptions about the signal and its stability that validation must examine.')
    table(['Task','Analysis'],[['1','Ridge under partial feature observation; interpolation and shrinkage'],['2','Lasso on the same dense-signal simulation'],['3','Chronological model selection and A/B/C return forecasts'],['4','Two-layer learning dynamics and a change in signal direction']],[75,420])

    title('1. Definitions and experimental controls')
    text('Signals in row t predict the next return: R<sub>t+1</sub> = S<sub>t</sub>′β* + ε<sub>t+1</sub>. The supplied CSV already contains this pairing, so the target column is not shifted. Let P be the true feature dimension, T<sub>tr</sub> the training size, P₁ the observed feature count, c=P/T<sub>tr</sub>, q=P₁/P and c<sup>q</sup>=P₁/T<sub>tr</sub>. In simulation Ψ=I, ε is standard normal and ‖β*‖²=b*.')
    text('Estimator and scaling',sub)
    text('β̂(z) = (X′X + T<sub>tr</sub> z I)<super>−1</super>X′y.<br/>Equivalently, minimize ‖y−Xβ‖²/T<sub>tr</sub> + z‖β‖². No intercept is fitted. At z=0 the SVD pseudoinverse gives the minimum-norm solution, including when P₁ exceeds T<sub>tr</sub>. A library Ridge implementation therefore needs alpha=T<sub>tr</sub>z; using alpha=z would change the experiment substantially.')
    text('Out-of-sample measurement',sub)
    text('For each test prediction ŷ, define the timing return g=ŷy. All expectations below are test-sample averages:<br/>R²<sub>paper</sub> = 1 − mean[(y−ŷ)²]/mean[y²] = (2 mean[ŷy]−mean[ŷ²])/mean[y²].<br/>E[g] = mean[ŷy]; SR = mean[g]/√mean[g²].<br/>SR<sub>var</sub> = mean[g]/√var(g), with population-convention sample variance (ddof=0).<br/>The fourth plotted quantity is ‖β̂‖². These definitions follow KMZ [1] and the assignment [3].')
    text('The R² benchmark is a zero forecast, not the sample mean. The primary Sharpe is not annualized and uses an uncentered second moment. Positive rescaling of predictions leaves either Sharpe unchanged but alters R² and expected timing return. Thus an almost-zero forecast can have a nonzero Sharpe while offering negligible expected return. An exactly zero strategy is assigned SR=0 by explicit implementation convention; its ratio is mathematically 0/0.')
    table(['Control','Specification'],[['Training / test / true features','200 / 5,000 / 2,000; c=10'],['Signal strength and repetitions','b*=1; 16 independent runs'],['cᵩ grid','0.5, 0.75, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 7, 10'],['Ridge z grid','0, 0.1, 0.5, 2, 10, 50, 100'],['Lasso α grid','0.01, 0.03, 0.1, 0.3'],['Randomness','config.json; NumPy Generator; one nested permutation per run']],[140,355])
    text('Code: src/core.py::metrics and ridge_path; src/simulation.py::run_simulations. Numerical tests verify the R² identity, Sharpe scaling, library penalty equivalence and rank-deficient minimum-norm behavior.',small)

    title('2. Task 1: ridge under partial observation')
    text('Each run draws Gaussian features, a dense Gaussian coefficient vector normalized to b*, and independent noise. Returns use all 2,000 true features. One random column permutation is fixed for the run; each larger observed block contains the smaller one. Training and test rows are disjoint. This isolates the benefit of revealing additional signals while holding the underlying return process and data split fixed.')
    figure('ridge_b1','Figure 1. Fixed-z ridge curves at b*=1, averaged over 16 independent runs. Shading denotes mean ±1.96 standard errors across runs, pointwise and descriptive. R² uses a symmetric-log axis and coefficient norm a log axis to retain the interpolation spike. The black dashed curve is the DGP-known ridge reference, not test-selected tuning. Source: simulation_raw.csv → src/figures.py::make_figures.')
    text(f'The interpolation region is unstable: at cᵩ=0.98 and 1.02, mean ridgeless R² is {mean(1,"ridge",0,.98,"r2"):.2f} and {mean(1,"ridge",0,1.02,"r2"):.2f}, respectively. Its coefficient norm grows sharply and Sharpe falls close to zero. Beyond interpolation, the minimum-norm solution becomes less variable and prediction improves, producing the second descent in error. Ridge damps the small singular values that cause the spike.')
    text('Negative R² does not imply a negative timing mean or Sharpe: the prediction magnitude can be excessive even when its direction contains useful information. Conversely, a well-controlled norm alone does not establish that a model has learned a signal.')

    title('3. Task 1: interpreting the ridge benchmark')
    figure('ridge_regularized','Figure 2. Regularized ridge R² at b*=1, focusing on z=2, 10 and 50 near and beyond interpolation. The black dashed curve is the DGP-known shrinkage reference. Shading shows pointwise mean ±1.96 standard errors across 16 runs. Source: simulation_raw.csv → src/figures.py::make_figures.')
    text('With identity covariance, observed and omitted blocks are uncorrelated, and the observed spectrum does not depend on q. Under the additional small cross-block trace condition, KMZ Proposition 6 gives z*(q)=c[1+b*(1−q)]/b*. Here b*=1 and c=10, so z*=20−10q. Because q=cᵩ/10, the DGP-known reference penalty falls from 19.5 at cᵩ=0.5 to 10 at cᵩ=10. The table shows selected points on that curve.')
    text('This is a theoretical reference that uses known simulation parameters; it is not estimated from test scores and cannot be copied directly to A/B/C, where the true signal is unknown. KMZ Theorem 1 concerns population R² and Sharpe under optimal shrinkage and its assumptions. It does not promise monotonic improvement for every fixed penalty or finite sample.')
    table(['Observed cᵩ','Revealed fraction q','DGP-known z*','Fixed-grid neighbor'],[[cq,f(cq/10),f(20-cq),10] for cq in [0.5,0.98,1.02,10]],[110,120,115,120])

    title('4. Task 2: Lasso on the identical experiment')
    text('Lasso minimizes ‖y−Xβ‖²/(2T<sub>tr</sub>)+α‖β‖₁, with no intercept. It uses exactly the same simulated rows, returns, nested feature blocks and complexity grid as ridge. All four penalties are fixed in config.json; none is chosen using simulated test outcomes. Its sparse representation supplies a meaningful alternative to the dense shrinkage of ridge.')
    figure('lasso_b1','Figure 3. Lasso at b*=1 on the same 16 runs as Figure 1. The four fixed-α curves show all required metrics; the black dashed line remains the DGP-known ridge reference. Shading is pointwise mean ±1.96 SE. Source: src/simulation.py::run_simulations → simulation_raw.csv → src/figures.py::make_figures.')
    text('The dense isotropic DGP distributes signal over many small coefficients. Lasso suppresses many of these coefficients and cannot be expected to inherit the ridge theorem. Weak penalties retain estimation noise; strong penalties approach a zero forecast and remove both noise and signal. Its interpolation region is smoother than the ridgeless ridge spike, but smoothness does not imply better prediction.')
    text(f'At cᵩ=10, fixed α=0.1 has R²={f(mean(1,"lasso",.1,10,"r2"))} and SR={f(mean(1,"lasso",.1,10,"sharpe"))}; ridge z=10 has R²={f(mean(1,"ridge",10,10,"r2"))} and SR={f(mean(1,"ridge",10,10,"sharpe"))}. This comparison identifies the cost of a sparsity bias in this DGP, not a general ranking of Lasso and ridge.')

    title('5. Task 2: penalty strength and limits')
    rows=[]
    for a in [.01,.03,.1,.3]:
        rows.append([a,f(mean(1,'lasso',a,10,'r2')),f(mean(1,'lasso',a,10,'sharpe')),f(mean(1,'lasso',a,10,'norm2'))])
    table(['α at cᵩ=10','R²','Paper SR','‖β̂‖²'],rows,[110,125,125,125])
    text('As α increases, Lasso shrinks more coefficients toward zero. At α=0.3, the coefficient norm is only 0.0070 and R² is close to zero from below: the forecast is nearly zero, so it approaches the zero-prediction benchmark. At α=0.01, the norm is larger and the model captures more of the dense signal, but it also retains more estimation noise. No α is selected from simulated test scores. A sparse true model could reverse this comparison, but this task deliberately uses a dense true coefficient vector. The A/B/C files do not disclose their coefficient structure, so those choices are assessed separately.')
    text('All Lasso paths converged without warnings. Their maximum recorded dual gap and per-run diagnostics are available in lasso_convergence.csv. The bands quantify variation across repeated datasets, not uncertainty about model misspecification in real markets.',small)

    title('6. Task 3: validation and model selection')
    table(['Pair','Training rows','Features','Test rows','P/T train'],[[p,decisions[p]['n_train'],decisions[p]['features'],decisions[p]['n_test'],f(decisions[p]['features']/decisions[p]['n_train'])] for p in 'ABC'],[45,105,95,110,140])
    text('All features and returns are finite. The t column contains strictly increasing integers. Feature order is obtained from the training header and enforced when reading test features; missing or additional columns cause an error. Input SHA-256 hashes preserve the identity of the supplied data. Test features are used only to construct final forecasts after training decisions have been made.')
    text('Two levels of chronological validation',sub)
    text('First, reserve the last 20% of each training file as an outer audit. Within the first 80%, use expanding fits on its first 40%, 60% and 80%, validating on the subsequent 20% blocks. Pool all inner validation predictions and select the trained candidate with the lowest mean squared error. Because the validation targets are fixed, this is equivalent to maximizing pooled paper R². Evaluate that complete selection procedure once on the outer segment.')
    text('For the A/B/C forecasts, run the identical expanding-window selection rule on the entire public training file and refit its selected candidate on all public training rows. The final choice may differ because it has more evidence. The outer audit evaluates the earlier selection pipeline, not the final fitted model. Final selection-CV scores are optimistic diagnostics after searching candidates; neither score is a hidden-test result. Outer outcomes were not used to redesign the search grid.')
    table(['Family','Fixed candidate grid'],[['Ridge','z = 0.1, 1, 5, 10, 25, 50, 100'],['Lasso','α = 0.01, 0.03, 0.1, 0.3'],['Elastic Net','α = 0.03, 0.1; L1 ratio = 0.2'],['RBF kernel ridge','γ = 0.1/P or 1/P; z = 0.1, 1, 10'],['Diagnostic baselines','Zero and fitting-sample historical mean; not selectable models']],[125,370])
    text('Each fit divides columns by fitting-only RMS values. This equalizes penalties while retaining a zero intercept; validation or test rows never determine scaling. Ridge and Lasso keep the objective conventions in Section 1. Kernel ridge uses K+T<sub>fit</sub>zI and γ proportional to 1/P; its RBF feature map represents a modest nonlinear alternative. No leverage multiplier is tuned after selection.')
    text('Code: src/prediction.py::candidate_predictions, rolling_select, run_predictions. Artifacts: validation_folds.csv, validation_ranking.csv, model_decisions.json and data_audit.csv.',small)

    title('7. Task 3: evidence, decisions and uncertainty')
    table(['Pair','Final model','Selection-CV R²','Outer-pipeline R²','Outer SR'],[[p,decisions[p]['final_selected'],f(decisions[p]['selection_cv_r2']),f(decisions[p]['outer_metrics']['r2']),f(decisions[p]['outer_metrics']['sharpe'])] for p in 'ABC'],[35,155,100,110,95])
    figure('prediction_validation','Figure 4. Best candidate within each family on final selection CV. These are model-search scores, not an unbiased comparison on unseen labels. Historical-mean and zero baselines are diagnostics. Source: validation_ranking.csv → src/figures.py::make_figures.')
    text('Pair A gives the clearest evidence: several moderately and heavily regularized ridge candidates yield positive validation R² and SR. The final selection is z=5. The earlier inner selection was z=10 and also achieved positive R² on the outer segment. Dense shrinkage is therefore defensible for A, while a claim that its true coefficients are dense would exceed the available evidence.')
    text('For B, even the best trained candidate has negative selection-CV R². The chosen heavily shrunk RBF forecast is close to zero. For C, the tiny positive RBF selection score is outweighed by the uncertainty suggested by the negative outer-pipeline score. In high-dimensional Gaussian-like features, an RBF kernel can be close to a constant off-diagonal matrix; these choices need not demonstrate useful nonlinear structure. Model selection here supplies a reproducible decision under weak evidence, not a discovery of a trading signal.')
    text('The search criterion prioritizes calibrated squared-error prediction. Sharpe and timing mean are reported but not separately optimized, avoiding an additional noisy search. Strong shrinkage may improve R² by reducing exposure without improving directional Sharpe. A different ex-ante grading utility could prefer a different model. With only 48 outer observations for B and 72 for C, sampling uncertainty is substantial.')
    text('Forecast files contain exactly t,yhat in UTF-8, retain public-test row order, and contain no missing or nonfinite predictions. Returns are in the original data units; no annualization or hidden-label evaluation is performed.',small)

    title('8. Task 4: what the neural-network paper shows')
    text('Montanari and Urbani [2] study a two-layer network f(x)=m<super>−1</super>Σ aᵢσ(wᵢ′x), with ‖wᵢ‖=1, trained by projected gradient flow under squared loss. Inputs are i.i.d. Gaussian, and the target depends on a low-dimensional latent projection plus independent noise. Initialization scale and the growth of second-layer weights are central. Their analysis connects representation learning and late overfitting within the same dynamical picture.')
    table(['Regime','Mean-field initialization: qualitative behavior'],[['Early, t=O(1)','Learn latent features; training and test risk fall together; second-layer L1 norm remains O(1).'],['Intermediate, 1≪t≪m','Extended feature learning; generalization gap remains small in the asymptotic setting; second-layer magnitude grows slowly.'],['Late, t on the order of m','Second-layer norm approaches order √m; training and test risk separate; latent alignment can decay and test risk increase.']],[135,360])
    text('The new insight is a separation of timescales: a sufficiently small-complexity initialization creates an opportunity to learn useful features before much slower noise fitting destroys part of that representation. Merely saying “early stopping regularizes” misses this mechanism. Lazy initialization, with much larger second-layer weights, can skip useful representation learning and reach an interpolator with poorer generalization.')
    text('Strength and limits of the theory',sub)
    text('The long-time picture relies on dynamical mean field theory for a Gaussian surrogate, asymptotic analysis and numerical evidence. It is not a universal finite-network theorem. The rigorous finite-sample bound in Section 3 implies a small generalization gap on a weaker scale, t̂=o((n/d)<super>1/4</super>), under boundedness and regularity assumptions; it does not prove every detail of the late-time O(m) scenario.')
    text('The statement that wider networks overfit later holds in a comparison with n/(md) fixed. Increasing m in that comparison also increases n. It cannot be transferred unchanged to a trading problem with a fixed historical window. Gaussian independent samples, simple two-layer structure and gradient flow also differ from dependent, heavy-tailed returns and adaptive optimization.')
    text('Financial question',sub)
    text('Can validation-based early stopping retain a learned return signal, and does that protection survive a change in the direction of the signal? This separates two errors: memorizing noise from an unchanged process, and accurately learning a relationship that later becomes stale. The second error is not corrected by choosing the stopping time well on the old regime.')

    title('9. Task 4: a controlled financial illustration')
    text('Generate R=1.5 tanh(u′X)+ε with X∼N(0,I₁₆), ‖u‖=1 and ε∼N(0,0.7²). Each of four seeds supplies 256 training, 256 validation and 2,000 test observations. A paired shifted test set reuses test inputs and noise but rotates u by 60 degrees toward an independently drawn orthogonal direction. Rotation keeps signal strength unchanged and isolates a change in predictability direction.')
    figure('neural_dynamics','Figure 5. Means over four independent seeds. Two widths (32 and 128) use the same sample sizes, initial aᵢ=1, tanh activations and normalized first-layer weights. Projected Euler updates use step 0.02 for 20,000 steps (scaled time 400); checkpoints are recorded every 200 steps. Test sets are evaluated for diagnosis, never for choosing the checkpoint. Source: src/dynamics.py::run_dynamics → dynamics_raw.csv → src/figures.py::make_figures.')
    text('The reported clock is t̂, for dynamics dθ/dt̂=−mP∇R̂, matching the rescaled convention discussed in [2]. Each finite step re-normalizes first-layer weights. The validation checkpoint minimizes stationary validation MSE over recorded checkpoints. The last checkpoint is a predeclared comparator. The same selected weights are evaluated in both stationary and shifted tests.')
    text('This is an illustrative finite-network experiment. It does not solve the DMFT equations, use the paper’s large-system ratios, or claim exact reproduction of its asymptotic scaling. The numerical step was reduced from an initial 0.1 after detecting late optimization jumps; data, seeds and time horizon were retained. The report uses only the corrected run.')

    title('10. Task 4: findings and financial interpretation')
    g=checkpoints.groupby(['width','policy','split'])[['mse','sharpe','alignment','output_l1']].mean()
    rows=[]
    for width in cfg['dynamics']['widths']:
        for policy in ['validation_stop','last']:
            stationary=g.loc[(width,policy,'stationary')];shift=g.loc[(width,policy,'shifted')]
            rows.append([width,'Early' if policy=='validation_stop' else 'Last',f(stationary.mse),f(shift.mse),f(stationary.sharpe),f(shift.sharpe)])
    table(['m','Checkpoint','MSE stable','MSE shifted','SR stable','SR shifted'],rows,[35,90,95,95,90,90])
    for width in cfg['dynamics']['widths']:
        early=g.loc[(width,'validation_stop','stationary')];late=g.loc[(width,'last','stationary')]
        text(f'For m={width}, validation stopping obtains stationary test MSE {early.mse:.4f}, compared with {late.mse:.4f} at the last checkpoint. Mean absolute alignment is {early.alignment:.3f} at the selected checkpoint and {late.alignment:.3f} late; mean |aᵢ| changes from {early.output_l1:.3f} to {late.output_l1:.3f}. These paired diagnostics support the proposed representation/complexity mechanism in this finite example.')
    text('The shifted-target evaluation adds a separate limitation. A checkpoint selected entirely in the old regime has much poorer shifted-test error even when it generalizes well to stationary observations. Early stopping controls the use of an existing dataset; it does not make the latent relationship invariant. Improvement in MSE also need not translate into the same ranking by financial SR after a regime change.')
    text('Independent implication',sub)
    text('A financial training policy should treat representation age and training duration as separate choices. A rolling-window experiment could jointly vary the history length and stopping checkpoint, using nested time splits to select both. An old but well-regularized representation may be less useful than a freshly estimated one. The appropriate practical question is therefore whether the prediction relationship remains valid during deployment, alongside whether optimization is memorizing noise.')
    text('A focused next test would introduce a gradual rotation over a sequence of regimes, compare expanding versus fixed rolling windows, and select checkpoints on the immediately subsequent validation block. Evaluation would use untouched future blocks with R², SR, timing mean, turnover and transaction costs. If rolling refits reduce shifted error while early stopping alone does not, that would support separate control of signal age. If the gain vanishes after costs or across seeds, the trading implication would remain unconfirmed.')
    text('Limits: only four seeds, one smooth target, independent Gaussian observations and one abrupt 60-degree shift are used. Reported means are descriptive. The alignment proxy is observable because the teacher is known; it is unavailable on real returns. The networks need not fully interpolate by the finite horizon, and decreasing alignment alone does not prove the paper’s exact feature-unlearning limit. No conclusion about real-market profitability follows.')

    title('11. Reproducibility and implementation')
    text('The experiments use relative paths, fixed random seeds, pinned dependencies and single-threaded numerical execution. Task 1/2 simulations, A/B/C model selection and forecasts, neural trajectories and figures are reproducible by running run.py --task all with the supplied input data. Input-file hashes identify the exact six CSVs used in the empirical analysis.')
    text('Source code: <link href="https://github.com/xuhengyi/return-prediction-and-generalization" color="#0072B2">github.com/xuhengyi/return-prediction-and-generalization</link>. Implementation paths in this report are relative to the repository root. The repository includes example configuration, dependency versions and numerical checks; input CSVs are supplied separately.',small)
    table(['Analysis','Implementation and evidence'],[['Figures 1–3; simulation metrics','src/simulation.py::run_simulations; simulation_raw.csv'],['Model scores and A/B/C forecasts','src/prediction.py::run_predictions; validation_ranking.csv; model_decisions.json'],['Figure 5 and checkpoint comparison','src/dynamics.py::run_dynamics; dynamics_raw.csv; dynamics_checkpoints.csv'],['PDF/PNG plots','src/figures.py::make_figures'],['Numerical and file checks','tests/; scripts/validate.py; validation_report.json']],[170,325])
    text('Environment and numerical checks',sub)
    text('Python 3.12; NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.1, scikit-learn 1.8.0, Matplotlib 3.10.0, PyTorch 2.10.0 (CPU execution). Unit tests check metric identities, zero-strategy convention, regularization scaling and rank-deficient pseudoinverses. Artifact validation checks prediction schema, row order, finite values, expected counts and reproducibility against a separate environment. Small floating-point differences across BLAS libraries are allowed; statistical conclusions should be stable.')
    text('References',sub)
    text('[1] Kelly, B., Malamud, S., and Zhou, K. (2024). The Virtue of Complexity in Return Prediction. <i>The Journal of Finance</i>, 79(1), 459–503. DOI: <link href="https://doi.org/10.1111/jofi.13298" color="#0072B2">10.1111/jofi.13298</link>. Journal issue year is 2024; retrieved CrossRef BibTeX carries the December 2023 online-publication date.',small)
    text('[2] Montanari, A., and Urbani, P. (2025). Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks. <i>Advances in Neural Information Processing Systems</i>, 38. DOI: <link href="https://doi.org/10.52202/085713-0895" color="#0072B2">10.52202/085713-0895</link>. Conference version supplied with the assignment; official citation saved in references/montanari_urbani.bib.',small)
    text('[3] DSA5205 (2026). Project 1: Reproduce Key Results in a ML-in-Finance Study. Assignment dated September 2, 2026, including Appendices A–B. Local file: DSA5205Fall2026Project1.pdf.',small)
    text('Implementation uses NumPy/SciPy numerical routines, scikit-learn estimators and PyTorch differentiation. The official KMZ replication program was not used as a substitute for finite-sample training.',small)
    def page(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C6D5DF'));canvas.line(50,43,545,43)
        canvas.setFont('Text',8);canvas.setFillColor(colors.HexColor('#5B7081'))
        canvas.drawString(50,29,f'DSA5205 · {student["id"]} · Complexity and generalization')
        canvas.drawRightString(545,29,str(doc.page))
    filename=output/f'{student["id"]}_report.pdf'
    doc=SimpleDocTemplate(str(filename),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=44,bottomMargin=55,title='Complexity, Shrinkage and Learning Time in Financial Return Prediction',author=student['name'])
    doc.build(story,onFirstPage=page,onLaterPages=page)
    (output/'report_content.md').write_text('\n\n'.join(markdown),encoding='utf-8')
    print(f'Report written: {filename}',flush=True)
