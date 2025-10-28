import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import tensorflow as tf
import tensorflow_ranking as tfr
from glob import glob

def intersection_over_union(bbox1, bbox2, eps=1e-6):
    """
    두 바운딩 박스 간의 IoU를 계산합니다.
    bbox 형식: {'l': left, 't': top, 'r': right, 'b': bottom}
    """
    l1, t1, r1, b1 = bbox1['l'], bbox1['t'], bbox1['r'], bbox1['b']
    l2, t2, r2, b2 = bbox2['l'], bbox2['t'], bbox2['r'], bbox2['b']
    
    # 교차 영역 계산
    left = max(l1, l2)
    right = min(r1, r2)
    bottom = max(b1, b2)
    top = min(t1, t2)
    
    # 교차 영역이 없으면 IoU는 0
    if right <= left or top <= bottom:
        return 0.0

    # 교차 영역 넓이
    intersection_area = (right - left) * (top - bottom)
    
    # 두 박스의 넓이
    area1 = (r1 - l1) * (t1 - b1)
    area2 = (r2 - l2) * (t2 - b2)
    
    # IoU 계산
    union_area = area1 + area2 - intersection_area
    return intersection_area / (union_area + eps)


def improved_bbox_matching(groundtruth_data, result_data, iou_threshold=0.2):
    """
    같은 페이지에 있는 GT와 Result 박스를 비교하여 IoU가 높은 순으로 매칭하는 함수.
    """
    all_matches = []
    
    for gt_idx, gt_item in enumerate(groundtruth_data):
        gt_bbox = gt_item['bbox']
        gt_page = gt_item['page']
        gt_id = gt_item['id']
        gt_class = gt_item.get('category_name')

        for result_idx, result in enumerate(result_data):
            for bbox_idx, result_item in enumerate(result.get('chunk_bboxes', [])):
                result_bbox = result_item.get('bbox')
                result_page = result_item.get('page')
                result_class = result_item.get('type')

                if result_bbox is None:
                    continue
                
                # 같은 페이지에 있는 경우에만 비교 수행
                if gt_page != result_page:
                    continue
                
                iou = intersection_over_union(gt_bbox, result_bbox)

                # IoU가 임계값 이상이면 유효한 매칭으로 간주
                if iou >= iou_threshold:
                    all_matches.append({
                        'gt_idx': gt_idx,
                        'result_idx': result_idx,
                        'bbox_idx': bbox_idx,
                        'gt_id': gt_id,
                        'gt_class': gt_class,
                        'result_class': result_class,
                        'iou': iou
                    })

    # IoU가 높은 순으로 정렬
    all_matches.sort(key=lambda x: x['iou'], reverse=True)

    # 각 GT와 Result 박스를 한 번씩만 매칭
    matched_gt = set()
    matched_result = set()
    final_matches = []

    for match in all_matches:
        gt_key = match['gt_idx']
        result_key = (match['result_idx'], match['bbox_idx'])

        if gt_key not in matched_gt and result_key not in matched_result:
            matched_gt.add(gt_key)
            matched_result.add(result_key)
            final_matches.append(match)
    
    # 클래스 정확도 계산
    correct_class_count = 0
    total_matches = len(final_matches)
    
    for match in final_matches:
        gt_class = match['gt_class']
        result_class = match['result_class']
        
        if gt_class == result_class:
            correct_class_count += 1
    
    class_accuracy = correct_class_count / total_matches if total_matches > 0 else 0

    return final_matches, class_accuracy


def calculate_f1_score(matches, groundtruth_data, result_data):
    """
    매칭 결과를 기반으로 F1 점수를 계산합니다.
    """
    TP = len(matches)
    
    total_predictions = 0
    for result in result_data:
        total_predictions += sum(1 for item in result.get('chunk_bboxes', []) if 'bbox' in item)
    
    FP = total_predictions - TP
    FN = len(groundtruth_data) - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def draw_all_boxes_on_pdf(pdf_path, groundtruth_data, result_data, matches, output_path):
    """
    모든 GT와 예측 박스를 PDF 위에 그려서 저장합니다.
    """
    # 매칭된 GT ID 목록
    matched_gt_ids = {match['gt_id'] for match in matches}
    
    # 매칭되지 않은 GT 찾기
    unmatched_gt = [gt for gt in groundtruth_data if gt['id'] not in matched_gt_ids]
    
    # 매칭된 예측 박스 인덱스
    matched_pred_indices = {(match['result_idx'], match['bbox_idx']) for match in matches}
    
    # 매칭되지 않은 예측 박스 찾기
    unmatched_pred = []
    for result_idx, result in enumerate(result_data):
        for bbox_idx, bbox_item in enumerate(result.get('chunk_bboxes', [])):
            if (result_idx, bbox_idx) not in matched_pred_indices and 'bbox' in bbox_item:
                unmatched_pred.append({
                    'result_idx': result_idx,
                    'bbox_idx': bbox_idx,
                    'bbox': bbox_item['bbox'],
                    'page': bbox_item['page'],
                    'class': bbox_item.get('type', 'Unknown')
                })
    
    # PDF 열기
    doc = fitz.open(pdf_path)
    
    # 바운딩 박스 좌표 변환 함수
    def convert_bbox(bbox, page_width, page_height):
        x0 = bbox['l'] * page_width
        y0 = page_height - (bbox['t'] * page_height)
        x1 = bbox['r'] * page_width
        y1 = page_height - (bbox['b'] * page_height)
        return [x0, y0, x1, y1]
    
    # 1. 매칭된 박스 그리기
    for match in matches:
        gt_idx = match['gt_idx']
        result_idx = match['result_idx']
        bbox_idx = match['bbox_idx']
        iou_value = match['iou']
        
        gt_bbox = groundtruth_data[gt_idx]['bbox']
        gt_page = groundtruth_data[gt_idx]['page']
        gt_class = match['gt_class']
        
        result_bbox = result_data[result_idx]['chunk_bboxes'][bbox_idx]['bbox']
        result_page = result_data[result_idx]['chunk_bboxes'][bbox_idx]['page']
        result_class = match['result_class']
        
        if gt_page != result_page:
            continue
        
        page = doc[gt_page - 1]  # PDF 페이지 가져오기
        
        # PDF 크기 가져오기
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 클래스 일치 여부에 따라 색상 결정
        is_class_match = gt_class == result_class
        
        # GT 박스 (초록색 - 클래스 일치, 빨간색 - 클래스 불일치)
        gt_rect = convert_bbox(gt_bbox, page_width, page_height)
        gt_color = (0, 0.5, 0) if is_class_match else (1, 0, 0)
        page.draw_rect(gt_rect, color=gt_color, width=2)
        page.insert_text((gt_rect[0], gt_rect[1] - 10), f"GT: {gt_class}", color=gt_color)
        
        # 예측 박스 (파란색)
        result_rect = convert_bbox(result_bbox, page_width, page_height)
        pred_color = (0, 0, 1)
        page.draw_rect(result_rect, color=pred_color, width=2)
        page.insert_text((result_rect[0], result_rect[1] - 20), 
                         f"Pred: {result_class} (IoU: {iou_value:.2f})", 
                         color=pred_color)
    
    # 2. 매칭되지 않은 GT 박스 그리기 (녹색)
    for gt in unmatched_gt:
        gt_bbox = gt['bbox']
        gt_page = gt['page']
        gt_class = gt.get('category_name', 'Unknown')
        
        page = doc[gt_page - 1]
        page_width = page.rect.width
        page_height = page.rect.height
        
        gt_rect = convert_bbox(gt_bbox, page_width, page_height)
        page.draw_rect(gt_rect, color=(0, 1, 0), width=2)
        page.insert_text((gt_rect[0], gt_rect[1] - 10), f"GT: {gt_class} (No Match)", color=(0, 1, 0))
    
    # 3. 매칭되지 않은 예측 박스 그리기 (회색)
    for pred in unmatched_pred:
        pred_bbox = pred['bbox']
        pred_page = pred['page']
        pred_class = pred['class']
        
        page = doc[pred_page - 1]
        page_width = page.rect.width
        page_height = page.rect.height
        
        pred_rect = convert_bbox(pred_bbox, page_width, page_height)
        page.draw_rect(pred_rect, color=(0.5, 0.5, 0.5), width=2)
        page.insert_text((pred_rect[0], pred_rect[1] - 10), f"Pred: {pred_class} (No Match)", color=(0.5, 0.5, 0.5))
    
    # PDF 저장
    doc.save(output_path)
    doc.close()
    
    return output_path


def draw_problem_boxes_on_pdf(pdf_path, groundtruth_data, result_data, matches, output_path, iou_threshold=0.5):
    """
    클래스가 일치하지 않거나 IoU가 낮은 박스만 PDF에 그려서 저장합니다.
    """
    # 문제가 있는 매칭 필터링 (클래스 불일치 또는 IoU가 낮은 경우)
    problem_matches = [match for match in matches if 
                      (match['gt_class'] != match['result_class']) or 
                      (match['iou'] <= iou_threshold)]
    
    if not problem_matches:
        print("클래스 불일치 또는 낮은 IoU 매칭이 없습니다.")
        return None
    
    # PDF 열기
    doc = fitz.open(pdf_path)
    
    for match in problem_matches:
        gt_idx = match['gt_idx']
        result_idx = match['result_idx']
        bbox_idx = match['bbox_idx']
        iou_value = match['iou']
        
        gt_bbox = groundtruth_data[gt_idx]['bbox']
        gt_page = groundtruth_data[gt_idx]['page']
        gt_class = match['gt_class']
        
        result_bbox = result_data[result_idx]['chunk_bboxes'][bbox_idx]['bbox']
        result_page = result_data[result_idx]['chunk_bboxes'][bbox_idx]['page']
        result_class = match['result_class']
        
        if gt_page != result_page:
            continue
        
        page = doc[gt_page - 1]  # PDF 페이지 가져오기
        
        # PDF 크기 가져오기
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 바운딩 박스 좌표 변환
        def convert_bbox(bbox):
            x0 = bbox['l'] * page_width
            y0 = page_height - (bbox['t'] * page_height)
            x1 = bbox['r'] * page_width
            y1 = page_height - (bbox['b'] * page_height)
            return [x0, y0, x1, y1]
        
        # 문제 유형 결정
        is_class_mismatch = gt_class != result_class
        is_low_iou = iou_value <= iou_threshold
        
        problem_type = []
        if is_class_mismatch:
            problem_type.append("Class Mismatch")
        if is_low_iou:
            problem_type.append(f"Low IoU: {iou_value:.2f}")
        
        problem_str = ", ".join(problem_type)
        
        # GT 박스 (빨간색)
        gt_rect = convert_bbox(gt_bbox)
        page.draw_rect(gt_rect, color=(1, 0, 0), width=2)
        page.insert_text((gt_rect[0], gt_rect[1] - 10), f"GT: {gt_class}", color=(1, 0, 0))
        
        # 예측 박스 (파란색)
        result_rect = convert_bbox(result_bbox)
        page.draw_rect(result_rect, color=(0, 0, 1), width=2)
        page.insert_text((result_rect[0], result_rect[1] - 20), 
                         f"Pred: {result_class} ({problem_str})", 
                         color=(0, 0, 1))
    
    # PDF 저장
    doc.save(output_path)
    doc.close()
    
    return output_path


def evaluate_document(gt_path, result_path, pdf_path, output_dir):
    """
    문서에 대한 평가를 수행하고 결과를 저장합니다.
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 추출
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # JSON 파일 읽기
    with open(gt_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # COCO 형식의 데이터를 정규화된 형식으로 변환
    groundtruth_data = []
    
    # 이미지 ID별 width, height 저장
    image_info = {img["id"]: (img["width"], img["height"]) for img in coco_data.get('images', [])}
    
    # 카테고리 ID와 이름 매핑
    category_mapping = {cat["id"]: cat["name"] for cat in coco_data.get('categories', [])}
    
    for annotation in coco_data.get('annotations', []):
        bbox = annotation['bbox']
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # 해당 이미지의 width, height 가져오기
        image_width, image_height = image_info.get(image_id, (None, None))
        
        if image_width is None or image_height is None:
            print(f"⚠ Warning: Image ID {image_id} not found in images data.")
            continue
        
        # COCO 형식의 bbox [x, y, width, height]를 정규화된 형식으로 변환
        x, y, w, h = bbox
        
        # 정규화된 바운딩 박스 계산
        l_normalized = x / image_width
        t_normalized = (image_height - y) / image_height
        r_normalized = (x + w) / image_width
        b_normalized = (image_height - (y + h)) / image_height
        
        normalized_bbox = {
            "l": l_normalized,
            "t": t_normalized,
            "r": r_normalized,
            "b": b_normalized,
            "coord_origin": "BOTTOMLEFT"
        }
        
        # 카테고리 이름 가져오기
        category_name = category_mapping.get(category_id, "Unknown")
        
        # 정규화된 데이터 추가
        groundtruth_data.append({
            "id": annotation['id'],
            "bbox": normalized_bbox,
            "page": image_id,  # 페이지 번호로 image_id 사용
            "category_id": category_id,
            "category_name": category_name
        })
    
    # 박스 매칭 및 클래스 정확도 계산
    
    matches, class_accuracy = improved_bbox_matching(groundtruth_data, result_data, iou_threshold=0.2)
    
    # F1 점수 계산
    f1_metrics = calculate_f1_score(matches, groundtruth_data, result_data)
    
    # 결과 저장
    output_matches = {match['gt_id']: {
        'groundtruth_id': match['gt_id'],
        'gt_class': match['gt_class'],
        'result_class': match['result_class'],
        'iou': match['iou'],
        'class_match': match['gt_class'] == match['result_class']
    } for match in matches}
    
    # 결과 JSON 저장
    results = {
        'filename': filename,
        'class_accuracy': class_accuracy,
        'precision': f1_metrics['precision'],
        'recall': f1_metrics['recall'],
        'f1_score': f1_metrics['f1_score'],
        'matches': output_matches
    }
        # IoU 통계 계산
    iou_values = [match['iou'] for match in matches]
    if iou_values:
        avg_iou = np.mean(iou_values)
        median_iou = np.median(iou_values)
        min_iou = min(iou_values)
        max_iou = max(iou_values)
        iou_50_ratio = np.mean(np.array(iou_values) >= 0.5)
        iou_75_ratio = np.mean(np.array(iou_values) >= 0.75)
    else:
        avg_iou = median_iou = min_iou = max_iou = iou_50_ratio = iou_75_ratio = 0

    # 결과 JSON에 IoU 통계 추가
    results['iou_stats'] = {
        'avg_iou': avg_iou,
        'median_iou': median_iou,
        'min_iou': min_iou,
        'max_iou': max_iou,
        'iou_50_ratio': iou_50_ratio,
        'iou_75_ratio': iou_75_ratio
    }

    result_json_path = os.path.join(output_dir, f"{filename}_evaluation.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 모든 박스 시각화
    all_boxes_pdf = os.path.join(output_dir, f"{filename}_all_boxes.pdf")
    draw_all_boxes_on_pdf(pdf_path, groundtruth_data, result_data, matches, all_boxes_pdf)
    
    # 문제 박스 시각화 (클래스 불일치 또는 IoU ≤ 0.5)
    problem_boxes_pdf = os.path.join(output_dir, f"{filename}_problem_boxes.pdf")
    draw_problem_boxes_on_pdf(pdf_path, groundtruth_data, result_data, matches, problem_boxes_pdf, iou_threshold=0.5)
    
    # 결과 출력
    print(f"===== {filename} 평가 결과 =====")
    print(f"클래스 정확도: {class_accuracy:.4f}")
    print(f"평균 IoU: {avg_iou:.4f}")
    print(f"결과 저장 경로: {output_dir}")
    print("=== 추가정보 ===")
    print(f"매칭된 박스 수: {len(matches)}")
    print(f"그라운드 트루스 박스 수: {len(groundtruth_data)}")
    print(f"예측 박스 수: {sum(len(result.get('chunk_bboxes', [])) for result in result_data)}")
    print(f"중앙값 IoU: {median_iou:.4f}")
    print(f"최소 IoU: {min_iou:.4f}")
    print(f"최대 IoU: {max_iou:.4f}")
    print(f"IoU >= 0.5 비율: {iou_50_ratio:.4f}")
    print(f"IoU >= 0.75 비율: {iou_75_ratio:.4f}")
    print(f"Precision: {f1_metrics['precision']:.4f}")
    print(f"Recall: {f1_metrics['recall']:.4f}")
    print(f"F1 Score: {f1_metrics['f1_score']:.4f}")  
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 문서의 바운딩 박스 평가 스크립트")
    parser.add_argument("--gt_path", required=True, help="Ground Truth JSON 파일 경로")
    parser.add_argument("--result_path", required=True, help="예측 결과 JSON 파일 경로")
    parser.add_argument("--pdf_path", required=True, help="원본 PDF 파일 경로")
    parser.add_argument("--output_dir", default="./evaluation_results", help="결과 저장 디렉토리 경로(선택)")
    args = parser.parse_args()
    
    # 입력 인자 가져오기
    gt_path = args.gt_path
    result_path = args.result_path
    pdf_path = args.pdf_path
    output_dir = args.output_dir
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 필요한 파일이 모두 존재하는지 확인
    if not os.path.exists(gt_path):
        print(f"오류: Ground Truth 파일이 존재하지 않습니다: {gt_path}")
        exit(1)
    
    if not os.path.exists(result_path):
        print(f"오류: 결과 파일이 존재하지 않습니다: {result_path}")
        exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"오류: PDF 파일이 존재하지 않습니다: {pdf_path}")
        exit(1)
    
    # 문서 평가 수행
    evaluate_document(gt_path, result_path, pdf_path, output_dir)
