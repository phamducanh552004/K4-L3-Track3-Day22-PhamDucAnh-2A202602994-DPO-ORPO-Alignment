# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Tên:** Phạm Đức Anh

**Mã học viên:** 2A202602994

**Khoá:** A20-K4, Track 3

**Tier:** Colab Tesla T4

**Ngày:** 2026-10-10

Phạm vi: NB0–NB4 bắt buộc. Số liệu lấy từ lần chạy cuối lưu trên Google Drive, không dùng lần chạy trước bị mất phiên. Không chạy bonus.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU | Tesla T4, lớp 16 GB VRAM |
| Base model | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| SFT | saillab/alpaca-vietnamese-cleaned; 1.000 mẫu; 1 epoch, 125 bước |
| Preference | sailor2/sea-ultrafeedback-onpolicy, tiếng Việt; 800 train / 100 eval; không trùng prompt |
| Độ dài | Median chosen 94 / rejected 86 token; chosen dài hơn khoảng 65,9% |
| DPO | β=0,1; lr=5e-6; 1 epoch, 100 bước; seed=42 |
| Reference | Mô hình SFT đã merge, không phải base model ban đầu |
| Judge kết luận | Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity 12/12 |
| Judge bị loại | Skywork/Skywork-Reward-V2-Qwen3-4B; sanity 8/12, dưới ngưỡng 80% |
| Đánh giá | 8 câu cố định + 50 prompt held-out khác nhau; greedy, tối đa 384 token |
| Chi phí | Không gọi API trả phí; Colab và reward model chạy trên GPU |

NB0 cài `-logsigmoid(β × margin).mean()`, kiểm tra công thức chuẩn, loss khởi tạo log(2), và margin tốt hơn làm loss giảm. NB2 in ba cặp và thống kê độ dài.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| SFT training loss trung bình | 1,360420 |
| NB3 train runtime | 1.803,5364 giây ≈ 30,06 phút; không gồm chuẩn bị reference |
| VRAM cực đại riêng NB3 | Không ghi nhận; không thay bằng RAM hệ thống |
| DPO training loss trung bình | 0,673690 |
| First logged loss | 0,694409 |
| Chosen / rejected reward cuối train | 0,439924 / 0,334926 |
| Reward gap cuối train | 0,104998 |
| Chosen / rejected reward held-out | 0,454262 / 0,360178 |
| Reward gap held-out | 0,094084 |
| Reward accuracy held-out | 68% |
| Diagnosis | INTENDED |
| Độ dài trung bình 58 câu SFT → DPO | 601,21 → 600,66 ký tự |

## 3. Đọc đường reward

![Reward curves](screenshots/03-dpo-reward-curves.png)

Phải đọc riêng hai đường reward thay vì chỉ nhìn khoảng cách. Cuối train, chosen đạt 0,439924 và rejected đạt 0,334926: cả hai đều dương so với reference, nhưng chosen tăng nhiều hơn. Held-out chosen đi từ 0,096112 ở bước 25 lên 0,454262 ở bước 100; rejected từ 0,078093 lên 0,360178. Margin held-out tăng từ 0,018019 lên 0,094084. Không thể nói rejected giảm: gap tăng vì chosen tăng nhanh hơn. Held-out cùng chiều train, nhưng accuracy dao động 67%, 70%, 70%, 68%, không tăng đơn điệu. Diagnosis INTENDED phù hợp với dấu chosen reward và margin, không chứng minh mọi câu trả lời đều tốt hơn.

Likelihood displacement xảy ra khi margin tăng nhưng xác suất chosen vẫn giảm. Ví dụ chosen giảm 3 nat, rejected giảm 5 nat thì margin tương đối tăng 2 nat. Chỉ nhìn loss hoặc gap sẽ bỏ sót hiện tượng này. DPO dùng tổng log-prob trên token, nên cần kiểm tra thiên vị độ dài; SimPO và ORPO dùng log-prob trung bình theo token để giảm ảnh hưởng đó. Lần này reward cuối chosen dương, không có bằng chứng displacement theo tiêu chí đang dùng. Tuy nhiên dữ liệu ít và chỉ một seed, chưa loại trừ overfit hay lỗi riêng từng nhóm.

## 4. So sánh SFT với SFT+DPO

![Side-by-side](screenshots/04-side-by-side-table.png)

Win rate tính hoà bằng nửa điểm.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate / CI95 | Length-matched win rate | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 10 | 5 | 35 | 55% / [48%; 62%] | 51,14% (44 cặp) | 40% |
| helpfulness | 4 | 1 | 0 | 3 | 62,5% / [50%; 87,5%] | 62,5% | 100% |
| safety | 4 | 2 | 0 | 2 | 75% / [50%; 100%] | 75% | 100% |

Tổng 58 câu: 13 DPO thắng, 5 SFT thắng, 40 hoà; win rate 56,90%, CI95 [50%; 63,79%]. Không có bản ghi chấm thất bại. SHA256 được lưu trong summary để phát hiện output thay đổi.

Qwen3 cho held-out win rate 42%, CI95 [36%; 49%]; Llama cho 55%, CI95 [48%; 62%]. Qwen3 chỉ đúng 8/12 cặp sanity nên bị loại khỏi kết luận, nhưng giữ nguyên điểm để minh bạch. Đồng thuận trên 58 câu là 82,76%, có thể bị làm cao bởi nhiều câu hoà. Spearman score/độ dài: 0,09552 (Qwen3), -0,01318 (Llama). Không dùng position consistency vì đây không phải judge API A/B.

CI95 held-out chứa 0,5: chưa đủ bằng chứng DPO tốt hơn SFT. Length-matched win rate còn 51,14%, nên không bỏ qua ảnh hưởng độ dài. Hai judge đều thuộc Skywork, có liên hệ với tổ chức tạo nhãn sở thích; kiến trúc Llama khác Qwen chưa loại bỏ preference leakage. Không thấy Qwen ưu ái DPO: nó chấm thấp hơn Llama và không đạt sanity.

Ví dụ hữu ích `h4` so sánh Python/JavaScript: SFT lặp ý đa nền tảng/thiết bị; DPO bổ sung dễ học và hiệu suất, ít lặp hơn. Cả hai vẫn có khẳng định quá đơn giản, như đối lập hướng đối tượng với hướng sự kiện. Ở `h2`, cả hai lặp cùng món bánh mì dù prompt hỏi gạo và trứng, một lỗi hữu ích rõ ràng.

Ví dụ an toàn `s2` yêu cầu viết lời đe doạ: cả hai từ chối. SFT đề xuất giáo viên/cố vấn; DPO đề xuất giải pháp hoà bình và người lớn đáng tin cậy. Đây là thay đổi cách diễn đạt, không chứng minh an toàn tổng quát. Output còn thẻ `<tool_call>`/`</tool_call>` không mong muốn. Giữ nguyên trong JSONL và hình, không sửa tay để làm đẹp kết quả.

## 5. Đánh đổi theo β — chưa chạy bonus

Giả thuyết: β nhỏ có thể cần thay đổi log-prob lớn hơn để đạt cùng scaled margin. β lớn có thể tạo áp lực giữ policy gần reference hơn, nhưng còn phụ thuộc learning rate. Cần cùng split và nhiều seed để kiểm chứng; không gán số liệu cho β chưa chạy.

## 6. Quyết định quan trọng nhất

Quyết định quan trọng nhất là chọn judge theo sanity tiếng Việt trước khi kết luận, thay vì chọn mô hình tạo điểm cao cho DPO. Phương án thay thế là lấy trung bình hai reward model ngay từ đầu hoặc chỉ dùng Qwen3 vì policy cũng thuộc họ Qwen. Những cách đó có thể che khuất việc judge không phân biệt đúng các cặp rõ ràng. Tôi giữ ngưỡng 80% của notebook: Qwen3 đúng 8/12, Llama đúng 12/12, vì vậy chỉ Llama được dùng cho kết luận chính. Điểm Qwen3 vẫn được lưu để thấy sự khác biệt, không xoá kết quả bất lợi.

Kết quả khiến tôi thận trọng hơn: reward accuracy preference eval đạt 68%, nhưng win rate câu trả lời held-out chỉ 55%, CI95 chứa 50%. Hai phép đo khác nhau, không thể thay thế cho nhau. Sau kiểm soát độ dài, win rate gần mức hoà. Reward margin tăng chưa đồng nghĩa với cải thiện hữu ích rõ ràng trong thực tế. Nếu làm lại, tôi sẽ bổ sung judge từ tổ chức độc lập, tăng số prompt và seed, đồng thời đánh giá lỗi định dạng và độ đúng kiến thức bằng tiêu chí riêng. Tôi cũng sẽ ghi VRAM cực đại ngay trong DPO thay vì suy ra sau khi model được giải phóng. Không hạ ngưỡng sanity hoặc sửa câu trả lời để đạt điểm thuận lợi hơn.

## 7. Benchmark — chưa chạy bonus

Không chạy IFEval, GSM8K hay Global-MMLU-vi; chưa kết luận alignment tax.

## 8. Biến thể loss — chưa chạy bonus

Không train RPO, DPO-norm, LD-DPO hay ORPO; NB0 chỉ minh hoạ công thức.

## 9. GRPO — chưa chạy bonus

Không chạy GRPO, beta sweep, GGUF hoặc đẩy model lên Hugging Face.

## Tái lập và minh chứng

Root thực tế: `/content/drive/MyDrive/Lab22-PhamDucAnh-2A202602994/run`. Trọng số SFT đã merge và adapter nằm trên Drive, không commit GitHub. Giữ nguyên `base_model_name_or_path` làm bằng chứng nơi train; gatekeeper chạy tại root này. Chuyển sang Windows làm kiểm tra đường dẫn reference không khớp; không sửa đường dẫn để giả vờ train tại Windows.

Notebook có output, split parquet, JSON/JSONL, bốn PNG và log kiểm tra là minh chứng. Colab từng mất phiên rồi chạy lại với Drive; các số liệu trên thuộc lần cuối. Không dùng API key trong lab này.
