# Phân Tích Hình Chương 3 - Mô Hình ARX(5,1,2)

File này liệt kê các hình trong notebook `ARX_512_V6 copy.ipynb` và gợi ý nội dung phân tích để đưa vào báo cáo. Nếu phần của bạn chỉ phụ trách ARX, nên ưu tiên dùng Fig. 1 đến Fig. 12. Các hình Fig. 13 đến Fig. 15 liên quan Kalman, MPC và toàn hệ thống, chỉ nên dùng nếu viết báo cáo chung của nhóm.

## Fig. 1 - Soil Moisture Và Setpoint Theo Thời Gian

**Nội dung hình:** Hình biểu diễn độ ẩm đất theo thời gian trên ba tập Train, Validation và Test, kèm theo ngưỡng setpoint thấp/cao.

**Phân tích:** Hình này dùng để mô tả dữ liệu đầu vào trước khi huấn luyện mô hình. Độ ẩm đất thay đổi theo thời gian do tác động của tưới, bay hơi và điều kiện môi trường. Việc chia dữ liệu theo thứ tự thời gian phù hợp với bài toán chuỗi thời gian, vì mô hình phải dự báo dữ liệu tương lai từ dữ liệu quá khứ. Nếu đường độ ẩm đất phần lớn nằm trong vùng setpoint, có thể nhận xét hệ thống vận hành tương đối ổn định. Những đoạn vượt ra khỏi vùng setpoint là các giai đoạn khó dự báo hơn.

## Fig. 2 - Phân Bố Các Biến Cảm Biến Và Điều Khiển

**Nội dung hình:** Hình histogram cho các biến `Soil_Moisture`, `Temperature`, `Humidity`, `Light`, `Drip`, `Mist` và `Fan`.

**Phân tích:** Hình này cho thấy các biến có thang đo khác nhau rõ rệt. Ví dụ, `Light` có miền giá trị lớn, trong khi `Drip`, `Mist`, `Fan` là biến bật/tắt. Điều này giải thích vì sao cần chuẩn hóa dữ liệu trước khi ước lượng mô hình. Phân bố giữa Train, Validation và Test cũng giúp kiểm tra dữ liệu test có khác biệt nhiều so với dữ liệu train hay không.

## Fig. 3a - FIT Theo Kiểu Dự Báo

**Nội dung hình:** So sánh chỉ số FIT giữa dự báo 1-step, 12-step và free-run trên Train, Validation và Test.

**Phân tích:** Mô hình ARX(5,1,2) đạt kết quả 1-step tốt hơn free-run. Trên tập test, FIT 1-step đạt khoảng **85.73%**, trong khi FIT free-run đạt khoảng **65.01%**. Điều này hợp lý vì dự báo 1-step dùng giá trị quá khứ thực tế, còn free-run dùng chính giá trị đã dự báo ở bước trước nên sai số bị tích lũy theo thời gian.

## Fig. 3b - RMSE Theo Kiểu Dự Báo

**Nội dung hình:** So sánh RMSE giữa dự báo 1-step, 12-step và free-run.

**Phân tích:** RMSE tăng khi chuyển từ 1-step sang 12-step và free-run. Đây là đặc điểm thường gặp trong mô phỏng chuỗi thời gian dài hạn. Trên tập test, RMSE free-run khoảng **1.009**, cho thấy sai số trung bình vẫn ở mức tương đối nhỏ so với miền biến thiên của độ ẩm đất.

## Fig. 4a - Validation FIT_sim Qua Các Version

**Nội dung hình:** So sánh FIT_sim trên tập Validation giữa các phiên bản ARX(5,1,2).

**Phân tích:** Hình này dùng để trình bày quá trình cải tiến mô hình. Các phiên bản sau lần lượt bổ sung chuẩn hóa, intercept, Ridge, feature mở rộng và clipping. Nếu FIT_sim tăng qua các phiên bản, có thể kết luận các cải tiến giúp mô hình mô phỏng ổn định hơn trên tập validation.

## Fig. 4b - Test FIT_sim Qua Các Version

**Nội dung hình:** So sánh FIT_sim trên tập Test giữa các phiên bản.

**Phân tích:** Đây là hình quan trọng để đánh giá khả năng tổng quát hóa. Với phiên bản V6, FIT_sim trên tập test đạt khoảng **65.01%**. Kết quả này cho thấy mô hình vẫn giữ được khả năng mô phỏng trên dữ liệu chưa dùng để huấn luyện hoặc chọn mô hình.

## Fig. 5a - FIT_sim Theo Alpha

**Nội dung hình:** Biểu diễn ảnh hưởng của hệ số Ridge `alpha` đến FIT_sim.

**Phân tích:** Hình này giải thích quá trình lựa chọn siêu tham số. Trong kết quả hiện tại, `best_alpha = 0.0`, tức nghiệm tốt nhất trong lưới thử tương ứng với OLS hoặc Ridge không phạt. Điều này cho thấy với dữ liệu và tập đặc trưng hiện tại, regularization chưa cải thiện free-run FIT so với nghiệm không phạt.

## Fig. 5b - Độ Lớn Vector Tham Số Theo Alpha

**Nội dung hình:** Thể hiện chuẩn L2 của vector tham số khi thay đổi `alpha`.

**Phân tích:** Khi `alpha` tăng, Ridge có xu hướng làm nhỏ các hệ số, giúp mô hình bớt nhạy với nhiễu và đa cộng tuyến. Hình này minh họa sự đánh đổi giữa độ khớp dữ liệu và độ ổn định tham số.

## Fig. 6 - Top 25 Tham Số Có Độ Lớn Lớn Nhất

**Nội dung hình:** Biểu đồ thanh thể hiện các hệ số có độ lớn lớn nhất trong mô hình.

**Phân tích:** Hình này giúp diễn giải mô hình ARX. Các hệ số trễ của `Soil_Moisture` thường có ảnh hưởng lớn vì độ ẩm đất có tính quán tính. Các hệ số của biến điều khiển và biến môi trường thể hiện tác động của tưới, ánh sáng, nhiệt độ và độ ẩm không khí lên độ ẩm đất. Do mô hình dùng z-score, độ lớn hệ số có thể được so sánh tương đối công bằng hơn giữa các biến.

## Fig. 7a - Validation Actual Vs Free-Run

**Nội dung hình:** Scatter plot giữa giá trị thực tế và giá trị dự báo free-run trên tập Validation.

**Phân tích:** Các điểm càng gần đường chéo thì dự báo càng chính xác. Nếu các điểm phân tán mạnh ở vùng giá trị cao hoặc thấp, mô hình có thể gặp khó ở các trạng thái cực trị. Hình này bổ sung cho biểu đồ theo thời gian vì nó cho thấy sai số theo miền giá trị.

## Fig. 7b - Test Actual Vs Free-Run

**Nội dung hình:** Scatter plot giữa giá trị thực tế và giá trị dự báo free-run trên tập Test.

**Phân tích:** Hình này phản ánh khả năng tổng quát hóa của mô hình. Nếu các điểm trên tập test vẫn bám quanh đường chéo, mô hình giữ được khả năng dự báo trên dữ liệu chưa thấy. Nếu có xu hướng lệch trên hoặc dưới đường chéo, mô hình có thể đang dự báo cao hơn hoặc thấp hơn thực tế ở một số vùng độ ẩm.

## Fig. 8a - Residual Free-Run Theo Thời Gian

**Nội dung hình:** Sai số dự báo free-run theo thời gian.

**Phân tích:** Residual dao động quanh 0 là dấu hiệu tốt, cho thấy mô hình không bị lệch hệ thống quá rõ. Nếu residual có các đoạn tăng hoặc giảm kéo dài, mô hình có thể chưa bắt hết động học trong những giai đoạn đó, ví dụ sau khi tưới hoặc khi điều kiện môi trường thay đổi nhanh.

## Fig. 8b - Phân Bố Residual Free-Run

**Nội dung hình:** Histogram của sai số dự báo free-run.

**Phân tích:** Nếu phân bố residual tập trung quanh 0, mô hình có sai số trung bình nhỏ. Nếu phân bố lệch trái hoặc lệch phải, mô hình có xu hướng dự báo cao hơn hoặc thấp hơn thực tế. Đuôi phân bố dài cho thấy tồn tại một số thời điểm sai số lớn.

## Fig. 8c - Residual Theo Giá Trị Dự Báo

**Nội dung hình:** Scatter plot residual theo giá trị dự báo.

**Phân tích:** Hình này kiểm tra sai số có phụ thuộc vào mức dự báo hay không. Nếu residual phân bố ngẫu nhiên quanh 0 trên toàn miền dự báo, mô hình tương đối ổn định. Nếu residual tạo thành xu hướng hoặc dạng hình phễu, có thể mô hình còn thiếu thành phần phi tuyến hoặc sai số thay đổi theo trạng thái.

## Fig. 8d - ACF Residual Test

**Nội dung hình:** Tự tương quan của residual trên tập Test.

**Phân tích:** Residual lý tưởng nên gần nhiễu trắng. Nếu ACF còn lớn ở nhiều lag, nghĩa là mô hình chưa khai thác hết thông tin động học trong chuỗi thời gian. Hình này giúp đánh giá bậc ARX(5,1,2) đã đủ hay cần tăng bậc trễ hoặc thêm đặc trưng.

## Fig. 9a - Impulse Response Top 8 Input

**Nội dung hình:** Đáp ứng xung của mô hình khi từng input thay đổi một đơn vị trong không gian chuẩn hóa.

**Phân tích:** Hình này cho thấy tác động của từng biến đầu vào lan truyền theo thời gian. Do mô hình có thành phần autoregressive, ảnh hưởng của input không chỉ xuất hiện tại một bước mà còn kéo dài qua nhiều bước tiếp theo. Các input có biên độ đáp ứng lớn là các biến có ảnh hưởng mạnh đến độ ẩm đất.

## Fig. 9b - Tổng Đáp Ứng 48 Bước

**Nội dung hình:** Tổng đáp ứng tích lũy trong 48 bước cho từng input.

**Phân tích:** Tổng đáp ứng dương nghĩa là input có xu hướng làm tăng độ ẩm đất theo mô hình; tổng đáp ứng âm nghĩa là input có xu hướng làm giảm độ ẩm. Hình này hữu ích để giải thích ý nghĩa vật lý của mô hình, ví dụ tưới thường có tác động dương, còn ánh sáng hoặc nhiệt độ có thể liên quan đến bay hơi.

## Fig. 10a - Thời Gian, Số Mẫu Và Missing Theo Split

**Nội dung hình:** Bảng thời gian bắt đầu/kết thúc, số mẫu và số giá trị thiếu của Train, Validation và Test.

**Phân tích:** Hình này dùng cho mục Dataset thực nghiệm. Nó chứng minh dữ liệu được chia theo thứ tự thời gian, phù hợp với bài toán dự báo chuỗi thời gian. Nếu số missing bằng 0, có thể ghi rằng dữ liệu không cần xử lý thiếu trước khi huấn luyện.

## Fig. 10b - Thống Kê Cơ Bản Các Biến

**Nội dung hình:** Bảng min, max, mean và std của các biến chính.

**Phân tích:** Bảng này mô tả miền hoạt động của từng biến cảm biến và điều khiển. Đây là cơ sở để giải thích lựa chọn z-score normalization, vì các biến có đơn vị và độ biến thiên khác nhau. Bảng cũng giúp người đọc hiểu điều kiện vận hành của hệ thống trong tập dữ liệu.

## Fig. 10c - Missing Data

**Nội dung hình:** Số lượng và tỷ lệ missing theo từng biến.

**Phân tích:** Hình này dùng để đánh giá chất lượng dữ liệu. Nếu missing bằng 0, dữ liệu có tính liên tục tốt và không cần nội suy. Nếu có missing, cần nêu rõ phương pháp xử lý như nội suy, forward-fill hoặc loại bỏ mẫu.

## Fig. 11a - ARX Vs Naive Baseline Theo FIT

**Nội dung hình:** So sánh FIT giữa mô hình ARX và baseline dự đoán bằng giá trị trước đó.

**Phân tích:** Baseline naive là mốc tham chiếu đơn giản nhưng quan trọng. Nếu ARX tốt hơn baseline, có thể kết luận mô hình không chỉ sao chép quán tính của chuỗi mà còn khai thác thêm thông tin từ input và cấu trúc trễ. Hình này làm phần đánh giá thuyết phục hơn so với chỉ báo cáo kết quả của riêng ARX.

## Fig. 11b - ARX Vs Naive Baseline Theo RMSE

**Nội dung hình:** So sánh RMSE giữa ARX và baseline naive.

**Phân tích:** RMSE thấp hơn nghĩa là dự báo chính xác hơn theo đơn vị độ ẩm đất. Hình này nên được dùng cùng Fig. 11a để trình bày cả mức độ khớp tương đối và sai số tuyệt đối. Nếu ARX giảm RMSE so với naive, đó là bằng chứng trực tiếp cho lợi ích của mô hình ARX.

## Fig. 12 - Phản Ứng Dự Báo Quanh Sự Kiện Tưới

**Nội dung hình:** So sánh đường thực tế và ARX free-run quanh các thời điểm `Drip` bật.

**Phân tích:** Đây là hình phân tích các trường hợp khó. Khi tưới bật, độ ẩm đất có thể thay đổi nhanh hơn trạng thái bình thường. Nếu mô hình bám được xu hướng sau tưới, chứng tỏ ARX đã học được tác động của input điều khiển. Nếu sai lệch lớn, có thể giải thích rằng động học sau tưới có tính phi tuyến hoặc phụ thuộc điều kiện đất/môi trường mà ARX tuyến tính chưa mô tả hết.

## Fig. 13a - Trước/Sau Lọc Kalman Trên Tập Test

**Nội dung hình:** So sánh tín hiệu raw `Soil_Moisture` với tín hiệu sau lọc Kalman.

**Phân tích:** Hình này thuộc phần Kalman, không phải phần ARX chính. Nếu báo cáo chung cần nhắc đến Kalman, có thể viết rằng Kalman làm mượt tín hiệu độ ẩm đất và giảm dao động ngắn hạn. Nếu bạn chỉ phụ trách ARX, không nên đưa hình này vào phần kết quả chính của bạn.

## Fig. 13b - Variance Của Sai Phân Tín Hiệu

**Nội dung hình:** So sánh variance của sai phân trước và sau lọc.

**Phân tích:** Variance của sai phân thấp hơn cho thấy tín hiệu sau lọc mượt hơn. Tuy nhiên, nếu làm mượt quá mạnh, bộ lọc có thể gây trễ và làm mất phản ứng nhanh của cảm biến.

## Fig. 13c - Innovation Của Adaptive Kalman

**Nội dung hình:** Innovation là sai khác giữa quan sát và dự báo trước cập nhật Kalman.

**Phân tích:** Innovation dao động quanh 0 cho thấy bộ lọc không bị lệch hệ thống rõ ràng. Các spike lớn thể hiện thời điểm tín hiệu thay đổi đột ngột hoặc nhiễu mạnh.

## Fig. 13d - R Thích Nghi Theo Thời Gian

**Nội dung hình:** Giá trị `R` của Adaptive Kalman thay đổi theo thời gian.

**Phân tích:** `R` đại diện cho mức nhiễu đo lường. Khi tín hiệu biến động mạnh, `R` có thể tăng để bộ lọc bớt tin vào quan sát tức thời. Hình này chỉ nên dùng nếu phần Kalman thuộc phạm vi báo cáo của nhóm.

## Fig. 14a - Độ Ẩm Đất Và Vùng Setpoint Trên Test

**Nội dung hình:** Đường độ ẩm đất thực tế và vùng setpoint thấp/cao.

**Phân tích:** Hình này mô tả điều kiện vận hành của dữ liệu. Với phần ARX, có thể dùng để giải thích rằng mô hình được huấn luyện trên dữ liệu có ràng buộc setpoint. Không nên trình bày hình này như kết quả MPC nếu bạn không phụ trách MPC.

## Fig. 14b - Hành Động Điều Khiển Ghi Nhận Trong Dữ Liệu

**Nội dung hình:** Tín hiệu `Drip`, `Mist`, `Fan` theo thời gian.

**Phân tích:** Hình này cho thấy input điều khiển là tín hiệu rời rạc/bật tắt. Các tín hiệu này là biến ngoại sinh trong mô hình ARX và là nguyên nhân làm trạng thái độ ẩm thay đổi.

## Fig. 14c - Tỷ Lệ Nằm Trong/Ngoài Dải Setpoint

**Nội dung hình:** Tỷ lệ thời gian độ ẩm đất nằm trong dải setpoint, dưới ngưỡng thấp hoặc trên ngưỡng cao.

**Phân tích:** Hình này liên quan nhiều đến đánh giá điều khiển. Nếu chỉ viết phần ARX, nên xem đây là thống kê dữ liệu vận hành, không phải kết quả MPC.

## Fig. 15 - Log Cần Thu Để Đánh Giá MPC Và Toàn Hệ Thống

**Nội dung hình:** Bảng liệt kê các loại log cần có để đánh giá latency, WebSocket, dashboard, pipeline AI và MPC.

**Phân tích:** Đây không phải kết quả thực nghiệm mà là khung đánh giá. Hình này chỉ nên đưa vào phần hạn chế hoặc hướng phát triển nếu báo cáo cần nói về những gì chưa đo được.

## Gợi Ý Chọn Hình Cho Phần ARX

Nếu báo cáo bị giới hạn số trang, nên ưu tiên các hình sau:

1. Fig. 1 - Dữ liệu Soil_Moisture theo thời gian.
2. Fig. 2 - Phân bố biến và lý do cần chuẩn hóa.
3. Fig. 3 - FIT và RMSE.
4. Fig. 4 - So sánh các version ARX.
5. Fig. 6 - Tham số quan trọng.
6. Fig. 7 - Actual vs predicted.
7. Fig. 8 - Residual diagnostics.
8. Fig. 9 - Impulse response.
9. Fig. 10 - Thống kê dataset.
10. Fig. 11 - So sánh với naive baseline.
11. Fig. 12 - Phân tích quanh sự kiện tưới.

Các hình Fig. 13 đến Fig. 15 không thuộc phần ARX chính, chỉ dùng nếu báo cáo chung của nhóm cần nhắc tới Kalman, MPC hoặc toàn hệ thống.
