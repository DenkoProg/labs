# Пояснення термінів та коду - Лабораторна робота 2

## Зміст

1. [Основні терміни](#основні-терміни)
2. [Алгоритми та методи](#алгоритми-та-методи)
3. [Детальний розбір коду](#детальний-розбір-коду)
4. [Математичні основи](#математичні-основи)

---

## Основні терміни

### Feature Detection (Виявлення ознак)

**Визначення**: Процес автоматичного знаходження характерних точок (features) на зображенні, які можна надійно виявити навіть при трансформаціях.

**Приклади**: кути, краї, blob-структури

**Чому важливо**: Ознаки використовуються для зіставлення зображень, розпізнавання об'єктів, 3D-реконструкції.

---

### Keypoint (Ключова точка)

**Визначення**: Характерна точка на зображенні з унікальними властивостями, яку можна надійно знайти на інших зображеннях того ж об'єкта.

**Властивості**:

- `pt` (x, y) - координати точки
- `size` - розмір околиці
- `angle` - орієнтація
- `response` - сила відгуку детектора

**У коді**:

```python
cv2.KeyPoint(float(x), float(y), self.patch_size, response=float(harris[y, x]))
```

---

### Descriptor (Дескриптор)

**Визначення**: Компактне числове представлення околиці ключової точки, яке дозволяє порівнювати різні точки між собою.

**Типи**:

- **Бінарні** (BRIEF, ORB) - послідовність бітів
- **Floating-point** (SIFT, SURF) - вектори дійсних чисел

**У коді**:

```python
descriptor = np.zeros(self.n_tests, dtype=np.uint8)  # 256 біт
```

---

### ORB (Oriented FAST and Rotated BRIEF)

**Визначення**: Швидкий та ефективний дескриптор, що поєднує детектор кутів з бінарним дескриптором.

**Компоненти**:

1. **FAST** (Features from Accelerated Segment Test) або **Harris Corner Detector**
2. **Orientation** - обчислення орієнтації через intensity centroid
3. **BRIEF** (Binary Robust Independent Elementary Features)

**Переваги**:

- Швидкий (∼100× швидше за SIFT)
- Rotation invariant (обертово-інваріантний)
- Безкоштовний (не захищений патентами)

---

### Harris Corner Detector

**Визначення**: Алгоритм виявлення кутів, що аналізує зміни інтенсивності в різних напрямках.

**Математика**:

```
M = [Iₓ²   IₓIᵧ]
    [IₓIᵧ  Iᵧ²  ]

R = det(M) - k·trace²(M)
  = λ₁λ₂ - k(λ₁ + λ₂)²
```

Де:

- `Iₓ`, `Iᵧ` - градієнти по x та y
- `λ₁`, `λ₂` - власні значення матриці M
- `k ≈ 0.04` - емпіричний коефіцієнт

**У коді**:

```python
def compute_harris_response(self, gray_img):
    Ix = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    Sxx = cv2.GaussianBlur(Ix * Ix, (5, 5), 0)
    Syy = cv2.GaussianBlur(Iy * Iy, (5, 5), 0)
    Sxy = cv2.GaussianBlur(Ix * Iy, (5, 5), 0)

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy

    return det - 0.04 * (trace**2)
```

**Інтерпретація**:

- Високий `R` → кут
- `R < 0` → край
- Малий `R` → плоска область

---

### Non-Maximum Suppression (NMS)

**Визначення**: Метод фільтрації, що залишає тільки локальні максимуми в межах околиці.

**Мета**: Уникнення скупчення ключових точок, вибір найсильніших.

**У коді**:

```python
kernel_size = 9
harris_dilated = cv2.dilate(harris, np.ones((kernel_size, kernel_size)))
keypoint_mask = (harris > threshold) & (harris == harris_dilated)
```

**Як працює**:

1. Дилатація розширює максимальні значення на 9×9 область
2. Порівнюємо оригінал з дилатованим
3. Точка залишається тільки якщо вона є локальним максимумом

---

### BRIEF (Binary Robust Independent Elementary Features)

**Визначення**: Бінарний дескриптор на основі порівняння пар пікселів.

**Алгоритм**:

1. Вибираємо N пар точок (x₁,y₁), (x₂,y₂) в патчі
2. Для кожної пари порівнюємо інтенсивності:
   ```
   τ(p; x₁,y₁, x₂,y₂) = 1 if I(x₁,y₁) < I(x₂,y₂) else 0
   ```
3. Дескриптор = конкатенація всіх бітів

**У коді**:

```python
for i, (x1, y1, x2, y2) in enumerate(rotated):
    px1, py1, px2, py2 = y + y1, x + x1, y + y2, x + x2

    if (0 <= px1 < gray_img.shape[0] and 0 <= py1 < gray_img.shape[1] and
        0 <= px2 < gray_img.shape[0] and 0 <= py2 < gray_img.shape[1]):
        descriptor[i] = 1 if gray_img[px1, py1] < gray_img[px2, py2] else 0
```

**Переваги**:

- Дуже швидкий (просто порівняння)
- Компактний (256 біт)
- Легко порівнювати (Hamming distance)

---

### Orientation (Орієнтація)

**Визначення**: Головний напрямок патчу, обчислений через розподіл інтенсивності.

**Метод Intensity Centroid**:

```
m₁₀ = Σ x·I(x,y)
m₀₁ = Σ y·I(x,y)

θ = atan2(m₀₁, m₁₀)
```

**У коді**:

```python
def compute_orientation(self, gray_img, keypoint):
    patch = gray_img[y-half_patch:y+half_patch+1,
                     x-half_patch:x+half_patch+1]

    m10 = m01 = 0.0
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            m10 += (j - half_patch) * patch[i, j]
            m01 += (i - half_patch) * patch[i, j]

    return np.arctan2(m01, m10)
```

**Навіщо**: Дозволяє обертати BRIEF патч у канонічну орієнтацію → rotation invariance.

---

### Matching (Зіставлення)

**Визначення**: Процес знаходження відповідностей між дескрипторами двох зображень.

**Типи**:

- **Brute-force** - порівняння кожного з кожним O(n²)
- **Approximate NN** - швидше, але менш точно (FLANN)

---

### Hamming Distance

**Визначення**: Кількість біт, у яких два бінарних вектори відрізняються.

**Формула**:

```
d_H(a, b) = popcount(a ⊕ b)
```

Де `⊕` - XOR, `popcount` - кількість одиниць.

**У коді**:

```python
def hamming_distance(self, desc1, desc2):
    return np.count_nonzero(desc1 != desc2)
```

**Приклад**:

```
a = 10110011
b = 11010001
      ↓ ↓  ↓
d_H = 3
```

---

### KNN Matching (K Nearest Neighbors)

**Визначення**: Для кожного дескриптора знаходимо k найближчих дескрипторів на другому зображенні.

**У коді**:

```python
def knn_match(self, desc1, desc2, k=2):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.array([self.hamming_distance(d1, d2) for d2 in desc2])
        k_nearest_indices = np.argpartition(distances, min(k, len(distances)-1))[:k]
        k_nearest_indices = k_nearest_indices[np.argsort(distances[k_nearest_indices])]

        k_matches = [cv2.DMatch(i, int(j), float(distances[j]))
                     for j in k_nearest_indices]
        matches.append(k_matches)

    return matches
```

**Зазвичай**: k=2 для Lowe's ratio test.

---

### Lowe's Ratio Test

**Визначення**: Метод фільтрації хибних відповідностей шляхом порівняння відстаней до найближчих двох сусідів.

**Формула**:

```
match is good if d₁ / d₂ < threshold
```

Де:

- `d₁` - відстань до найближчого сусіда
- `d₂` - відстань до другого найближчого
- `threshold ≈ 0.7-0.8`

**У коді**:

```python
def match(self, desc1, desc2):
    knn_matches = self.knn_match(desc1, desc2, k=2)
    good_matches = []

    for matches in knn_matches:
        if len(matches) >= 2:
            m, n = matches[0], matches[1]
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches
```

**Інтуїція**: Якщо найкраще співпадіння набагато краще за друге → надійне.

---

### DMatch

**Визначення**: Структура даних OpenCV для представлення відповідності.

**Поля**:

- `queryIdx` - індекс дескриптора на першому зображенні
- `trainIdx` - індекс дескриптора на другому зображенні
- `distance` - відстань між дескрипторами

**У коді**:

```python
cv2.DMatch(i, int(j), float(distances[j]))
```

---

### RANSAC (Random Sample Consensus)

**Визначення**: Ітеративний алгоритм для робастної оцінки параметрів моделі в присутності outliers.

**Алгоритм**:

```
1. Повторюємо max_iterations разів:
   a. Випадково вибираємо мінімальну множину точок
   b. Обчислюємо модель на цих точках
   c. Підраховуємо inliers (точки, що відповідають моделі)
   d. Якщо inliers більше за попередні → зберігаємо модель

2. Перераховуємо модель на всіх inliers найкращої ітерації
```

**У коді**:

```python
def find_transform(self, src_pts, dst_pts):
    best_M = None
    max_inlier_count = 0

    for _ in range(self.max_iterations):
        # Вибираємо 3 випадкові точки
        indices = np.random.choice(len(src_pts), 3, replace=False)
        M = self.estimate_affine_transform(src_pts[indices], dst_pts[indices])

        # Рахуємо inliers
        inliers, inlier_mask = self.count_inliers(src_pts, dst_pts, M)

        if len(inliers) > max_inlier_count:
            max_inlier_count = len(inliers)
            # Уточнюємо модель на всіх inliers
            M_refined = self.estimate_affine_transform(
                src_pts[inliers], dst_pts[inliers]
            )
            if M_refined is not None:
                best_M = M_refined

    return best_M, best_inlier_mask, params
```

**Параметри**:

- `max_iterations` - кількість ітерацій (1000-2000)
- `threshold` - максимальна відстань для inlier (3-5 пікселів)
- `min_inliers` - мінімальна кількість inliers для прийняття моделі

---

### Inliers vs Outliers

**Inliers** - точки, що добре вписуються в модель (< threshold)
**Outliers** - викиди, помилкові відповідності (≥ threshold)

**Візуалізація**:

- 🟢 Зелені лінії - inliers (правильні відповідності)
- 🔴 Червоні лінії - outliers (помилкові відповідності)

---

### Affine Transformation (Афінна трансформація)

**Визначення**: Лінійне перетворення, що зберігає паралельність ліній.

**Матриця 2×3**:

```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
                  [1]
```

**Включає**:

- Обертання (rotation)
- Масштабування (scale)
- Зміщення (translation)
- Зсув (shear)

**У коді**:

```python
def estimate_affine_transform(self, src_pts, dst_pts):
    A = []
    b = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0])
        b.append(x_prime)
        A.append([0, 0, 0, x, y, 1])
        b.append(y_prime)

    params = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]]])
```

**Екстракція параметрів**:

```python
def extract_params(self, M):
    tx, ty = M[0, 2], M[1, 2]
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

    angle = np.degrees(np.arctan2(c, a))
    scale = (np.sqrt(a**2 + c**2) + np.sqrt(b**2 + d**2)) / 2

    return {"angle": angle, "tx": tx, "ty": ty, "scale": scale}
```

---

### FLANN (Fast Library for Approximate Nearest Neighbors)

**Визначення**: Бібліотека для швидкого approximate nearest neighbor search.

**Для бінарних дескрипторів**: LSH (Locality-Sensitive Hashing)

**У коді**:

```python
FLANN_INDEX_LSH = 6
flann_matcher = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_LSH,
         table_number=12,      # кількість хеш-таблиць
         key_size=20,          # довжина ключа
         multi_probe_level=2), # рівень multi-probing
    dict(checks=100)           # кількість перевірок
)
```

**Переваги**: O(log n) замість O(n) для brute-force.

---

## Детальний розбір коду

### 1. CustomORB Class

#### Ініціалізація

```python
def __init__(self, n_features=1000, patch_size=31, n_tests=256):
    self.n_features = n_features      # кількість keypoints
    self.patch_size = patch_size      # розмір патчу (31×31)
    self.n_tests = n_tests            # кількість тестів BRIEF (256)

    np.random.seed(42)
    half_patch = patch_size // 2
    # Генеруємо випадкові пари точок для BRIEF
    self.test_points = np.random.randint(
        -half_patch + 2, half_patch - 2, size=(n_tests, 4)
    )
```

**Чому +2 та -2?** Щоб уникнути країв патчу при поворотах.

---

#### Виявлення ключових точок

```python
def detect_keypoints(self, gray_img):
    harris = self.compute_harris_response(gray_img)

    # Non-maximum suppression
    kernel_size = 9
    harris_dilated = cv2.dilate(harris, np.ones((kernel_size, kernel_size)))

    # Поріг і маска локальних максимумів
    threshold = 0.001 * harris.max()
    keypoint_mask = (harris > threshold) & (harris == harris_dilated)
    keypoints_coords = np.argwhere(keypoint_mask)

    # Вибираємо top N за response
    if len(keypoints_coords) > self.n_features:
        responses = harris[keypoints_coords[:, 0], keypoints_coords[:, 1]]
        top_indices = np.argsort(responses)[-self.n_features:]
        keypoints_coords = keypoints_coords[top_indices]

    return [cv2.KeyPoint(float(x), float(y), self.patch_size,
                         response=float(harris[y, x]))
            for y, x in keypoints_coords]
```

**Важливі моменти**:

1. NMS з kernel 9×9 для кращого розподілу точок
2. Threshold = 0.1% від максимального response
3. Сортування за response для вибору найкращих

---

#### Обчислення дескриптора з поворотом

```python
def compute_descriptor(self, gray_img, keypoint, angle):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    half_patch = self.patch_size // 2

    # Перевірка меж
    if (y - half_patch < 0 or y + half_patch >= gray_img.shape[0] or
        x - half_patch < 0 or x + half_patch >= gray_img.shape[1]):
        return None

    # Поворот test points на кут angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = np.zeros_like(self.test_points, dtype=np.int32)
    rotated[:, 0] = cos_a * self.test_points[:, 0] - sin_a * self.test_points[:, 1]
    rotated[:, 1] = sin_a * self.test_points[:, 0] + cos_a * self.test_points[:, 1]
    rotated[:, 2] = cos_a * self.test_points[:, 2] - sin_a * self.test_points[:, 3]
    rotated[:, 3] = sin_a * self.test_points[:, 2] + cos_a * self.test_points[:, 3]

    # Обчислюємо BRIEF
    descriptor = np.zeros(self.n_tests, dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(rotated):
        px1, py1 = y + y1, x + x1
        px2, py2 = y + y2, x + x2

        if (0 <= px1 < gray_img.shape[0] and 0 <= py1 < gray_img.shape[1] and
            0 <= px2 < gray_img.shape[0] and 0 <= py2 < gray_img.shape[1]):
            descriptor[i] = 1 if gray_img[px1, py1] < gray_img[px2, py2] else 0

    return descriptor
```

**Rotation matrix**:

```
[cos θ  -sin θ]
[sin θ   cos θ]
```

---

### 2. CustomMatcher Class

```python
class CustomMatcher:
    def __init__(self, ratio_threshold=0.75):
        self.ratio_threshold = ratio_threshold  # Lowe's ratio
```

**ratio_threshold**:

- 0.7 - strict (менше matches, більше точності)
- 0.75 - balanced
- 0.8 - relaxed (більше matches, менше точності)

---

### 3. CustomRANSAC Class

#### Підрахунок inliers

```python
def count_inliers(self, src_pts, dst_pts, M):
    # Застосовуємо трансформацію
    pts_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
    transformed = pts_homogeneous @ M.T

    # Обчислюємо Euclidean distance
    distances = np.linalg.norm(transformed - dst_pts, axis=1)

    # Точки в межах threshold - inliers
    inlier_mask = distances < self.threshold
    return np.where(inlier_mask)[0], inlier_mask
```

**Чому Euclidean distance?** Вимірюємо, наскільки точка після трансформації близька до очікуваної позиції.

---

### 4. Візуалізація відповідностей

```python
def draw_matches_custom(img1, kp1, img2, kp2, matches, inlier_mask=None, max_matches=50):
    # Створюємо панораму
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = max(h1, h2), w1 + w2

    panorama = np.zeros((h, w, 3), dtype=np.uint8)
    panorama[0:h1, 0:w1] = img1
    panorama[0:h2, w1:w1+w2] = img2

    # Малюємо лінії між відповідностями
    for i, match in enumerate(display_matches):
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = (int(kp2[match.trainIdx].pt[0]) + w1,
               int(kp2[match.trainIdx].pt[1]))

        # Колір залежить від inlier/outlier
        color = ((0, 255, 0) if display_mask is not None and display_mask[i]
                 else (255, 0, 0) if display_mask is not None
                 else (0, 255, 255))

        cv2.line(panorama, pt1, pt2, color, 2)      # жирніша лінія
        cv2.circle(panorama, pt1, 4, color, -1)     # більші кружечки
        cv2.circle(panorama, pt2, 4, color, -1)

    return panorama
```

---

## Математичні основи

### 1. Harris Corner Response

**Структурна матриця**:

```
M = Σ w(x,y) [Iₓ²   IₓIᵧ]
             [IₓIᵧ  Iᵧ²  ]
```

Де `w(x,y)` - Gaussian window

**Corner response**:

```
R = det(M) - k·trace²(M)
  = λ₁λ₂ - k(λ₁ + λ₂)²
```

**Інтерпретація власних значень**:

- λ₁ ≈ λ₂ ≈ 0 → flat region
- λ₁ >> λ₂ (or λ₂ >> λ₁) → edge
- λ₁ ≈ λ₂, both large → corner

---

### 2. Least Squares для Affine Transform

**Система рівнянь**:

```
x' = ax + by + tx
y' = cx + dy + ty
```

**Матрична форма** (для N точок):

```
[x₁  y₁  1  0   0   0 ] [a ]   [x'₁]
[0   0   0  x₁  y₁  1 ] [b ]   [y'₁]
[x₂  y₂  1  0   0   0 ] [tx]   [x'₂]
[0   0   0  x₂  y₂  1 ] [c ] = [y'₂]
[...               ... ] [d ]   [...]
                         [ty]
```

**Розв'язок**: `params = (AᵀA)⁻¹Aᵀb` або `np.linalg.lstsq(A, b)`

---

### 3. Rotation Angle з Affine Matrix

```
M = [a  b  tx]   [s·cos θ  -s·sin θ  tx]
    [c  d  ty] = [s·sin θ   s·cos θ  ty]
```

З першого стовпця:

```
a = s·cos θ
c = s·sin θ

θ = atan2(c, a)
s = √(a² + c²)
```

---

## Порівняння Custom vs OpenCV

### Точність

| Метрика           | Custom    | OpenCV   |
| ----------------- | --------- | -------- |
| Keypoints         | ~1000     | ~1000    |
| Harris threshold  | 0.001·max | Adaptive |
| NMS kernel        | 9×9       | Adaptive |
| BRIEF tests       | 256       | 256      |
| Ratio test        | 0.75      | 0.75     |
| RANSAC iterations | 2000      | 1000     |

### Швидкодія

- **Detection**: OpenCV ~2-3× швидше (оптимізований C++)
- **Matching**: FLANN ~5-10× швидше за brute-force
- **RANSAC**: Приблизно однакові

### Переваги Custom

✅ Повний контроль над параметрами
✅ Розуміння алгоритмів
✅ Можливість кастомізації

### Переваги OpenCV

✅ Оптимізований код
✅ Scale-space pyramid
✅ Adaptive thresholding
✅ Production-ready

---

## Практичні поради

### Налаштування параметрів

**n_features** (1000):

- Більше → краще coverage, але повільніше
- Менше → швидше, але може пропустити важливі features

**ratio_threshold** (0.75):

- Менше (0.7) → менше matches, більше precision
- Більше (0.8) → більше matches, менше precision

**RANSAC threshold** (5.0 пікселів):

- Менше (3.0) → строгіша модель, менше inliers
- Більше (7.0) → м'якша модель, більше inliers

**RANSAC iterations** (2000):

- Більше → вища вірогідність знаходження найкращої моделі
- Формула: `k = log(1-p) / log(1-w³)` де p=0.99, w=inlier_ratio

### Типові проблеми

**Мало matches**:

- ✅ Збільшити n_features
- ✅ Relaxed ratio_threshold (0.8)
- ✅ Перевірити якість зображень

**RANSAC fails**:

- ✅ Збільшити threshold (5.0 → 7.0)
- ✅ Зменшити min_inliers
- ✅ Більше iterations

**Повільна робота**:

- ✅ Зменшити n_features
- ✅ Використати FLANN замість brute-force
- ✅ Зменшити RANSAC iterations

---

## Висновки

### Ключові концепції

1. **Feature detection** - знаходження характерних точок
2. **Descriptor** - компактне представлення околиці
3. **Matching** - зіставлення дескрипторів
4. **RANSAC** - робастна оцінка моделі

### Pipeline ORB matching

```
Image → Grayscale → Harris corners → Top N keypoints
     → Orientation → Rotated BRIEF → Descriptors

Descriptors₁ + Descriptors₂ → Hamming distance → KNN
     → Lowe's ratio test → Good matches

Matches → RANSAC → Affine transform + Inliers
```

### Практичне застосування

- 📸 Image stitching (панорами)
- 🎯 Object recognition
- 📹 Visual odometry
- 🎮 AR/VR tracking
- 🤖 SLAM (Simultaneous Localization and Mapping)

---

_Документ створено для Лабораторної роботи 2_
_Курс: Обробка зображень методами штучного інтелекту_
_Національний університет «Львівська політехніка»_
