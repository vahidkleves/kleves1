import os
import re
import csv
import math
import ast
from typing import Any, Dict, Tuple, Optional
from bs4 import BeautifulSoup

# ------------------------- Configuration -------------------------
# You can override these with environment variables or CLI later if needed
HTML_FOLDER = os.environ.get("INS_HTML_DIR", "/workspace")
OUTPUT_CSV = os.environ.get("INS_OUTPUT_CSV", "/workspace/insulation_data.csv")

# ----------------------- Safe math evaluator ---------------------

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Mod,
    ast.FloorDiv,
    ast.LShift,
    ast.RShift,
)


def _safe_eval_expr(expr: str, variables: Dict[str, float]) -> float:
    """
    Safely evaluate a simple math expression using AST with variables.
    Supports +, -, *, /, **, unary +/-, parentheses and variables like T.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):  # for Python < 3.8
            return float(n.n)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Unsupported constant in expression")
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero in expression")
                return left / right
            if isinstance(n.op, ast.Pow):
                return left ** right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError("Division by zero in expression")
                return math.floor(left / right)
            if isinstance(n.op, ast.LShift):
                return float(int(left) << int(right))
            if isinstance(n.op, ast.RShift):
                return float(int(left) >> int(right))
            raise ValueError("Unsupported binary operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(n, ast.Name):
            if n.id in variables:
                return float(variables[n.id])
            raise ValueError(f"Unknown variable '{n.id}' in expression")
        raise ValueError("Unsupported expression element")

    # Validate allowed nodes
    for subnode in ast.walk(node):
        if not isinstance(subnode, _ALLOWED_AST_NODES):
            raise ValueError("Disallowed expression element in formula")

    return _eval(node)


def evaluate_k_formula(k_text: str, temperature_c: float) -> Optional[float]:
    """
    Evaluate the k(T) expression found in the library table at the given temperature.
    - Replaces ^ with **, '×' with '*', and uses variable T for temperature in Celsius.
    - If the text is a plain number, returns it as float.
    Returns None on failure.
    """
    if not k_text:
        return None

    expr = k_text.strip()

    # Quick path: simple numeric value
    try:
        return float(expr)
    except ValueError:
        pass

    # Normalize unicode and formatting issues
    expr = expr.replace("×", "*")
    expr = expr.replace("·", "*")
    expr = expr.replace("^", "**")
    expr = expr.replace("T^", "T**")  # sometimes T^2 present explicitly

    # Remove commas or stray characters that might appear
    expr = expr.replace(",", "")

    # Some sources put 10**(-7) as 10**-7, both are fine
    # Ensure balanced parentheses by best-effort: if unmatched, do nothing (AST will error)

    try:
        value = _safe_eval_expr(expr, {"T": float(temperature_c)})
        return float(value)
    except Exception:
        # Last resort: try to extract the first float as a fallback constant
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", k_text)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None


def calculate_heat_transfer_and_surface_temp(
    inner_temp: str, ambient_temp: str, air_speed: str, r_conv: str, r_cond: str
) -> Tuple[Any, Any]:
    """
    Calculate heat transfer (Q) and surface temperature (Ts).

    Q = (T_inner - T_ambient) / (R_conv + R_cond)
    Ts = T_ambient + Q * R_conv
    """
    try:
        T_inner = float(inner_temp)
        T_ambient = float(ambient_temp)
        _ = float(air_speed)  # currently unused; R_conv already accounts for it
        R_conv_val = float(r_conv)
        R_cond_val = float(r_cond)

        Q = (T_inner - T_ambient) / (R_conv_val + R_cond_val)
        Ts = T_ambient + Q * R_conv_val
        return round(Q, 6), round(Ts, 3)
    except (ValueError, ZeroDivisionError):
        return "N/A", "N/A"


# --------------------------- CSV header --------------------------
CSV_HEADER = [
    "file_name",
    "Inner Surface Temperature",
    "Ambient Temperature",
    "Air Speed",
    "Area",
    "R conv [C/W]",
    "Layer abb",
    "Layer index",
    "Layer Thickness_mm",
    "R cond [C/W]",
    "Material Name",
    "k (formula)",
    "k(T_mean)",
    "Density",
    "MaxTemp",
    "Heat Transfer Rate (Q)",
    "Surface Temperature (Ts)",
]


def parse_html_file(file_path: str) -> Tuple[list, list]:
    """
    Parse a single HTML file and return rows for CSV and a list of errors.
    """
    rows: list = []
    errors: list = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # --- Base info from summary tables ---
        info_data: Dict[str, str] = {}

        # Many templates have two summary tables with ids table_0 and table_1
        for div_id in ("table_0", "table_1"):
            container = soup.find("div", {"id": div_id})
            if container:
                table = container.find("table")
                if table:
                    for tr in table.find_all("tr")[1:]:
                        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"]) if c]
                        if cells and len(cells) >= 2:
                            info_data[cells[0]] = cells[1]

        # --- Insulation library ---
        ins_lib_container = soup.find("div", {"id": "Ins_Lib"})
        ins_lib_by_abb: Dict[str, Dict[str, str]] = {}
        ins_lib_by_index: Dict[str, Dict[str, str]] = {}
        if ins_lib_container:
            table = ins_lib_container.find("table")
            if table:
                for tr in table.find_all("tr")[1:]:
                    cells = [c.get_text(strip=True) for c in tr.find_all("td")]
                    if not cells or len(cells) < 10:
                        continue
                    item = {
                        "index": cells[0],
                        "code": cells[1],
                        "abb": cells[2],
                        "name": cells[3],
                        "density": cells[4],
                        "Thickness": cells[5],
                        "k": cells[6],
                        "Price": cells[7],
                        "Cur": cells[8] if len(cells) > 8 else "",
                        "MaxTemp": cells[9] if len(cells) > 9 else "",
                    }
                    ins_lib_by_abb[item["abb"]] = item
                    ins_lib_by_index[item["index"]] = item

        # --- Best choices table with layers ---
        best_container = soup.find("div", {"id": "table_2"})
        if best_container:
            best_table = best_container.find("table")
        else:
            best_table = None

        if not best_table:
            # No layers table; nothing to add
            return rows, errors

        # Extract context values used for computations
        inner_temp = info_data.get("Inner Surface Temperature", info_data.get("Internal Temperature", "0"))
        ambient_temp = info_data.get("Ambient Temperature", "0")
        air_speed = info_data.get("Air Speed", info_data.get("Wind Speed", "0"))
        r_conv = info_data.get("R conv [C/W]", info_data.get("R_conv [C/W]", "0"))
        area_val = info_data.get("Area [m2]", info_data.get("Total Surface Area", ""))

        # Iterate all rows (skip header)
        for tr in best_table.find_all("tr")[1:]:
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if not cells:
                continue

            # Expected order: abb | layer index | thickness_mm | R_cond | ...
            layer_abb = cells[0] if len(cells) > 0 else ""
            layer_index = cells[1] if len(cells) > 1 else ""
            layer_thickness = cells[2] if len(cells) > 2 else ""
            r_cond = cells[3] if len(cells) > 3 else "0"

            # Compute heat transfer and surface temperature for this configuration
            heat_transfer, surface_temp = calculate_heat_transfer_and_surface_temp(
                inner_temp, ambient_temp, air_speed, r_conv, r_cond
            )

            # Mean temperature used to evaluate k
            try:
                T_inner = float(inner_temp)
                Ts_val = float(surface_temp) if isinstance(surface_temp, (int, float)) else float(str(surface_temp))
                T_mean = (T_inner + Ts_val) / 2.0
            except Exception:
                # fallback to average with ambient if Ts not numeric
                try:
                    T_inner = float(inner_temp)
                    T_amb = float(ambient_temp)
                    T_mean = (T_inner + T_amb) / 2.0
                except Exception:
                    T_mean = None

            # Map library info by abb first, then by index as fallback
            lib_item = ins_lib_by_abb.get(layer_abb) or ins_lib_by_index.get(layer_index) or {}
            material_name = lib_item.get("name", "")
            k_formula = lib_item.get("k", "")
            density = lib_item.get("density", "")
            max_temp = lib_item.get("MaxTemp", "")

            # Compute k(T_mean)
            if T_mean is not None and k_formula:
                k_value = evaluate_k_formula(k_formula, T_mean)
            else:
                k_value = None

            rows.append([
                os.path.basename(file_path),
                inner_temp,
                ambient_temp,
                air_speed,
                area_val,
                r_conv,
                layer_abb,
                layer_index,
                layer_thickness,
                r_cond,
                material_name,
                k_formula,
                (round(k_value, 8) if isinstance(k_value, (int, float)) else "N/A"),
                density,
                max_temp,
                heat_transfer,
                surface_temp,
            ])

        return rows, errors

    except Exception as exc:
        errors.append(f"Failed to parse {os.path.basename(file_path)}: {exc}")
        return rows, errors


def main() -> None:
    rows: list = []
    errors: list = []

    for name in os.listdir(HTML_FOLDER):
        if not name.lower().endswith(".html"):
            continue
        fp = os.path.join(HTML_FOLDER, name)
        file_rows, file_errors = parse_html_file(fp)
        rows.extend(file_rows)
        errors.extend(file_errors)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"The information was extracted and saved in: '{OUTPUT_CSV}'")
    if errors:
        print("Warnings/Errors:")
        for e in errors:
            print("- ", e)


if __name__ == "__main__":
    main()