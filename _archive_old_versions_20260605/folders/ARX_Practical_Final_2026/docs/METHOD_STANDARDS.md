# Method Standards

Day la rule set dung cho benchmark cuoi.

## Tai lieu tham chieu

1. MathWorks System Identification Toolbox
   - URL: https://www.mathworks.com/products/sysid.html
   - Diem dung: dynamic model tu input-output data, linear/nonlinear model, compare/validate on test data, deployment/control design.

2. MathWorks `arx`
   - URL: https://www.mathworks.com/help/ident/ref/arx.html
   - Diem dung: ARX estimate bang least squares, cau truc `na`, `nb`, `nk`, fit percent/FPE/AIC/BIC.

3. MathWorks Validate Nonlinear ARX Models
   - URL: https://www.mathworks.com/help/ident/ug/validating-nonlinear-arx-models.html
   - Diem dung: validate bang data rieng, compare predicted/simulated output, residual analysis, khong dua ket luan chi tu training fit.

4. MathWorks NARX using `idnlarx`
   - URL: https://www.mathworks.com/help/ident/ug/estimating-narx-networks-using-idnlarx-instead-of-narxnet.html
   - Diem dung: NARX la nonlinear ARX, dung past y va lagged exogenous input; can danh gia ca open-loop/closed-loop.

5. Rawlings, Mayne, Diehl - Model Predictive Control: Theory, Computation, and Design
   - URL: https://sites.engineering.ucsb.edu/~jbraw/mpc/
   - Diem dung: MPC can plant model de predict future trajectory va optimize input sequence; neu model phi tuyen thi bai toan thanh nonlinear MPC/local linearization.

## Checklist dung logic thuc te

- Data split theo thoi gian: train -> validation -> test.
- Khong shuffle time-series.
- Standardization chi hoc tren train.
- Hyperparameter/order chon bang validation.
- Test set khong duoc dung de chon model.
- Tach ro:
  - `FIT_1step`: co true previous output, de dep nhung khong du dai han.
  - `FIT_12`: rolling multi-step, gan voi horizon ngan.
  - `FIT_sim`: free-run, nghiem khac nhat cho model dua vao MPC/simulation.
- Residual va data audit:
  - missing/duplicate/sampling irregular.
  - actuator policy: neu command sinh tu feedback output, day la closed-loop data, khong phai open-loop identification ly tuong.
  - neu future actuator chua duoc biet truoc, khong dung negative lag trong prediction.
- MPC:
  - ARX tuyen tinh phu hop LTI/adaptive MPC va RLS hon.
  - NARX/NNARX khong thay truc tiep ARX trong MPC tuyen tinh.
  - Neu dung NARX de dieu khien can NMPC, local linearization, hoac gain scheduling.

