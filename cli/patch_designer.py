import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox


MM_TO_PX = 3.0      # 1 mm -> 像素比例
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800
CANVAS_MARGIN = 20


class Electrode:
    def __init__(self, eid, x, y, diameter=5.0,
                 etype="measurement", group=""):
        self.id = str(eid)
        self.x = float(x)            # mm
        self.y = float(y)            # mm
        self.diameter = float(diameter)  # mm
        self.etype = etype           # "measurement" / "reference"
        self.group = group

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "diameter_mm": self.diameter,
            "type": self.etype,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d.get("id", ""),
            d.get("x", 0.0),
            d.get("y", 0.0),
            d.get("diameter_mm", 5.0),
            d.get("type", "measurement"),
            d.get("group", ""),
        )


class BoardOutline:
    def __init__(self, vertices=None):
        # vertices: list of (x_mm, y_mm)
        self.vertices = vertices or []

    def to_dict(self):
        if not self.vertices:
            return {"type": "none"}
        return {
            "type": "polygon",
            "vertices_mm": self.vertices,
            "closed": True,
        }

    @classmethod
    def from_dict(cls, d):
        if not d or d.get("type", "none") == "none":
            return cls([])
        verts = d.get("vertices_mm", [])
        return cls(verts)


class ElectrodeDesignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("电极贴可视化设计器")
        self.root.geometry("1200x820")

        # 数据
        self.electrodes = []
        self.selected_electrode_indices = []  # 索引列表
        self.board_outline = BoardOutline([])
        self.channel_count = 64  # 当前导联数

        # 板框绘制状态
        self.drawing_board_polygon = False
        self.temp_polygon_vertices = []  # 正在绘制的多边形
        self.selected_vertex_index = None

        # 拖动状态
        self.drag_mode = None          # None / "electrode" / "vertex"
        self.drag_electrode_index = None
        self.drag_vertex_index = None
        self.drag_start_world = None   # (x_mm, y_mm)
        self.drag_origin_pos = None    # 起始对象坐标 (x_mm, y_mm)
        self.drag_indicator_tag = "drag_indicator"

        # 测量和参考点
        self.measure_indicator_tag = "measure_indicator"
        self.ref_point_A = None        # (x_mm, y_mm)
        self.ref_point_tag = "ref_point_marker"

        # UI
        self._build_menu()
        self._build_main_ui()
        self._bind_canvas_events()

        # 默认新建 16 导联
        self._new_default_layout(16)

    # === UI 构建 ===

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="新建 16 导联", command=lambda: self._new_default_layout(16)
        )
        file_menu.add_command(
            label="新建 32 导联", command=lambda: self._new_default_layout(32)
        )
        file_menu.add_command(
            label="新建 64 导联", command=lambda: self._new_default_layout(64)
        )
        file_menu.add_command(
            label="新建 128 导联", command=lambda: self._new_default_layout(128)
        )
        file_menu.add_separator()
        file_menu.add_command(label="打开", command=self.load_from_json)
        file_menu.add_command(label="保存", command=self.save_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        board_menu = tk.Menu(menubar, tearoff=0)
        board_menu.add_command(label="矩形板框", command=self.create_rect_board)
        board_menu.add_command(label="绘制多边形板框", command=self.start_draw_board_polygon)
        board_menu.add_command(label="清除板框", command=self.clear_board)
        menubar.add_cascade(label="板框", menu=board_menu)

        layout_menu = tk.Menu(menubar, tearoff=0)
        layout_menu.add_command(label="线性布局（选中电极）", command=self.linear_layout_selected)
        menubar.add_cascade(label="布局", menu=layout_menu)

        self.root.config(menu=menubar)

    def _build_main_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # 左侧画布
        canvas_frame = tk.Frame(main_pane)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="white"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        main_pane.add(canvas_frame, stretch="always")

        # 右侧属性面板
        prop_frame = tk.Frame(main_pane, padx=5, pady=5)
        main_pane.add(prop_frame, width=380)

        # ---- 电极属性 ----
        lf_elec = ttk.LabelFrame(prop_frame, text="电极属性（对选中电极生效）")
        lf_elec.pack(fill=tk.X, pady=5)

        self.var_sel_info = tk.StringVar(value="导联总数：0，当前选中：0 个电极")
        ttk.Label(lf_elec, textvariable=self.var_sel_info).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(lf_elec, text="X (mm):").grid(row=1, column=0, sticky="e")
        ttk.Label(lf_elec, text="Y (mm):").grid(row=2, column=0, sticky="e")
        ttk.Label(lf_elec, text="直径 (mm):").grid(row=3, column=0, sticky="e")
        ttk.Label(lf_elec, text="类型:").grid(row=4, column=0, sticky="e")
        ttk.Label(lf_elec, text="组名:").grid(row=5, column=0, sticky="e")

        self.var_x = tk.StringVar()
        self.var_y = tk.StringVar()
        self.var_d = tk.StringVar()
        self.var_type = tk.StringVar(value="measurement")
        self.var_group = tk.StringVar()

        e_x = ttk.Entry(lf_elec, textvariable=self.var_x, width=12)
        e_y = ttk.Entry(lf_elec, textvariable=self.var_y, width=12)
        e_d = ttk.Entry(lf_elec, textvariable=self.var_d, width=12)
        e_x.grid(row=1, column=1, sticky="w")
        e_y.grid(row=2, column=1, sticky="w")
        e_d.grid(row=3, column=1, sticky="w")

        cmb_type = ttk.Combobox(
            lf_elec,
            textvariable=self.var_type,
            values=["measurement", "reference"],
            width=10,
            state="readonly",
        )
        cmb_type.grid(row=4, column=1, sticky="w")

        e_group = ttk.Entry(lf_elec, textvariable=self.var_group, width=12)
        e_group.grid(row=5, column=1, sticky="w")

        btn_apply_elec = ttk.Button(
            lf_elec, text="应用到选中电极", command=self.apply_electrode_properties
        )
        btn_apply_elec.grid(row=6, column=0, columnspan=2, pady=4)

        # ---- 分组操作 ----
        lf_group = ttk.LabelFrame(prop_frame, text="分组操作")
        lf_group.pack(fill=tk.X, pady=5)

        ttk.Label(lf_group, text="组名:").grid(row=0, column=0, sticky="e")
        self.var_group_name = tk.StringVar()
        e_grp_name = ttk.Entry(lf_group, textvariable=self.var_group_name, width=15)
        e_grp_name.grid(row=0, column=1, sticky="w")

        btn_assign_group = ttk.Button(
            lf_group, text="将选中电极设置为此组", command=self.assign_group_to_selected
        )
        btn_assign_group.grid(row=1, column=0, columnspan=2, pady=2, sticky="we")

        ttk.Label(lf_group, text="ΔX (mm):").grid(row=2, column=0, sticky="e")
        ttk.Label(lf_group, text="ΔY (mm):").grid(row=3, column=0, sticky="e")
        self.var_dx = tk.StringVar(value="0.0")
        self.var_dy = tk.StringVar(value="0.0")
        e_dx = ttk.Entry(lf_group, textvariable=self.var_dx, width=10)
        e_dy = ttk.Entry(lf_group, textvariable=self.var_dy, width=10)
        e_dx.grid(row=2, column=1, sticky="w")
        e_dy.grid(row=3, column=1, sticky="w")

        btn_move_group = ttk.Button(
            lf_group, text="平移整组电极", command=self.translate_group
        )
        btn_move_group.grid(row=4, column=0, columnspan=2, pady=2, sticky="we")

        # ---- 线性布局 ----
        lf_layout = ttk.LabelFrame(prop_frame, text="线性布局（对选中电极）")
        lf_layout.pack(fill=tk.X, pady=5)

        ttk.Label(lf_layout, text="起点 X (mm):").grid(row=0, column=0, sticky="e")
        ttk.Label(lf_layout, text="起点 Y (mm):").grid(row=1, column=0, sticky="e")
        ttk.Label(lf_layout, text="间距 (mm):").grid(row=2, column=0, sticky="e")
        ttk.Label(lf_layout, text="方向:").grid(row=3, column=0, sticky="e")

        self.var_lx = tk.StringVar(value="10.0")
        self.var_ly = tk.StringVar(value="10.0")
        self.var_lsp = tk.StringVar(value="5.0")
        self.var_ldir = tk.StringVar(value="horizontal")

        ttk.Entry(lf_layout, textvariable=self.var_lx, width=10).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Entry(lf_layout, textvariable=self.var_ly, width=10).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Entry(lf_layout, textvariable=self.var_lsp, width=10).grid(
            row=2, column=1, sticky="w"
        )

        cmb_dir = ttk.Combobox(
            lf_layout,
            textvariable=self.var_ldir,
            values=["horizontal", "vertical"],
            width=10,
            state="readonly",
        )
        cmb_dir.grid(row=3, column=1, sticky="w")

        ttk.Button(
            lf_layout, text="应用线性布局", command=self.linear_layout_selected
        ).grid(row=4, column=0, columnspan=2, pady=4, sticky="we")

        # ---- 板框属性 ----
        lf_board = ttk.LabelFrame(prop_frame, text="板框（多边形）")
        lf_board.pack(fill=tk.X, pady=5)

        ttk.Button(lf_board, text="矩形板框", command=self.create_rect_board).grid(
            row=0, column=0, columnspan=2, sticky="we", pady=2
        )
        ttk.Button(
            lf_board, text="绘制多边形板框", command=self.start_draw_board_polygon
        ).grid(row=1, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(lf_board, text="清除板框", command=self.clear_board).grid(
            row=2, column=0, columnspan=2, sticky="we", pady=2
        )

        ttk.Separator(lf_board, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, sticky="we", pady=4
        )

        ttk.Label(lf_board, text="选中顶点索引:").grid(row=4, column=0, sticky="e")
        self.var_vertex_idx = tk.StringVar(value="-")
        ttk.Label(lf_board, textvariable=self.var_vertex_idx).grid(
            row=4, column=1, sticky="w"
        )

        ttk.Label(lf_board, text="顶点 X (mm):").grid(row=5, column=0, sticky="e")
        ttk.Label(lf_board, text="顶点 Y (mm):").grid(row=6, column=0, sticky="e")
        self.var_vx = tk.StringVar()
        self.var_vy = tk.StringVar()
        ttk.Entry(lf_board, textvariable=self.var_vx, width=10).grid(
            row=5, column=1, sticky="w"
        )
        ttk.Entry(lf_board, textvariable=self.var_vy, width=10).grid(
            row=6, column=1, sticky="w"
        )

        ttk.Button(
            lf_board, text="应用顶点坐标", command=self.apply_vertex_coord
        ).grid(row=7, column=0, columnspan=2, sticky="we", pady=4)

        # 状态栏
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.var_status).pack(side=tk.LEFT)

    def _bind_canvas_events(self):
        # 普通左键
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-1>", self.on_canvas_double_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        # Alt + 左键：单独绑定为设置参考点 A，避免用 state 位判断
        self.canvas.bind("<Alt-ButtonPress-1>", self.on_canvas_alt_press)

    # === 坐标转换 ===

    def world_to_canvas(self, x_mm, y_mm):
        x_px = CANVAS_MARGIN + x_mm * MM_TO_PX
        y_px = CANVAS_HEIGHT - (CANVAS_MARGIN + y_mm * MM_TO_PX)
        return x_px, y_px

    def canvas_to_world(self, x_px, y_px):
        x_mm = (x_px - CANVAS_MARGIN) / MM_TO_PX
        y_mm = (CANVAS_HEIGHT - y_px - CANVAS_MARGIN) / MM_TO_PX
        return x_mm, y_mm

    # === 新建布局（16/32/64/128） ===

    def _new_default_layout(self, count):
        if self.electrodes:
            if not messagebox.askyesno(
                "确认", f"丢弃当前设计并新建 {count} 导联布局？"
            ):
                return

        self.channel_count = count
        self.electrodes = []
        self.selected_electrode_indices = []
        self.board_outline = BoardOutline([])
        self.temp_polygon_vertices = []
        self.drawing_board_polygon = False
        self.selected_vertex_index = None
        self.drag_mode = None
        self.ref_point_A = None
        self.canvas.delete(self.drag_indicator_tag)
        self.canvas.delete(self.measure_indicator_tag)
        self.canvas.delete(self.ref_point_tag)

        # 尽量接近正方形的网格 （rows * cols = count）
        rows = int(math.sqrt(count))
        while rows > 1 and count % rows != 0:
            rows -= 1
        cols = count // rows

        spacing = 10.0
        start_x, start_y = 20.0, 20.0
        eid = 1
        for r in range(rows):
            for c in range(cols):
                if eid > count:
                    break
                x = start_x + c * spacing
                y = start_y + r * spacing
                self.electrodes.append(
                    Electrode(eid, x, y, diameter=5.0, etype="measurement")
                )
                eid += 1

        self.update_selection_info()
        self.redraw()
        self.var_status.set(f"已新建 {count} 导联默认网格布局（{rows} x {cols}）")

    # === 绘图 ===

    def redraw(self):
        self.canvas.delete("all")
        self._draw_board_outline()
        self._draw_electrodes()
        self._draw_temp_polygon()
        # 参考点 A 也要重新画
        if self.ref_point_A is not None:
            self._draw_ref_point_A()

    def _draw_electrodes(self):
        for idx, elec in enumerate(self.electrodes):
            cx, cy = self.world_to_canvas(elec.x, elec.y)
            r_px = elec.diameter * 0.5 * MM_TO_PX
            color = "dodgerblue" if elec.etype == "measurement" else "orange"
            outline = "red" if idx in self.selected_electrode_indices else "black"
            width = 3 if idx in self.selected_electrode_indices else 1
            self.canvas.create_oval(
                cx - r_px,
                cy - r_px,
                cx + r_px,
                cy + r_px,
                fill=color,
                outline=outline,
                width=width,
            )
            self.canvas.create_text(cx, cy, text=elec.id, fill="white")

    def _draw_board_outline(self):
        if self.board_outline and self.board_outline.vertices:
            points = []
            for x, y in self.board_outline.vertices:
                cx, cy = self.world_to_canvas(x, y)
                points.extend([cx, cy])
            self.canvas.create_polygon(
                *points,
                outline="gray",
                fill="",
                dash=(4, 4),
                width=2
            )
            # 顶点控制点
            for i, (x, y) in enumerate(self.board_outline.vertices):
                cx, cy = self.world_to_canvas(x, y)
                size = 4
                self.canvas.create_rectangle(
                    cx - size,
                    cy - size,
                    cx + size,
                    cy + size,
                    fill="white",
                    outline="black",
                )

    def _draw_temp_polygon(self):
        if self.drawing_board_polygon and self.temp_polygon_vertices:
            pts = []
            for x, y in self.temp_polygon_vertices:
                cx, cy = self.world_to_canvas(x, y)
                pts.extend([cx, cy])
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill="green", width=2)
            for x, y in self.temp_polygon_vertices:
                cx, cy = self.world_to_canvas(x, y)
                self.canvas.create_oval(
                    cx - 3,
                    cy - 3,
                    cx + 3,
                    cy + 3,
                    outline="green",
                    fill="green",
                )

    # === 画布交互：按下/拖动/释放/移动 ===

    def on_canvas_press(self, event):
        wx, wy = self.canvas_to_world(event.x, event.y)

        # 正在绘制多边形板框：点击就是加点
        if self.drawing_board_polygon:
            self.temp_polygon_vertices.append((wx, wy))
            self.redraw()
            return

        # 重置拖动状态 & 清除拖动指示
        self.drag_mode = None
        self.drag_electrode_index = None
        self.drag_vertex_index = None
        self.drag_start_world = None
        self.drag_origin_pos = None
        self.canvas.delete(self.drag_indicator_tag)

        # 先尝试命中板框顶点（拖动顶点）
        if self.board_outline and self.board_outline.vertices:
            vidx = self._hit_test_vertex(event.x, event.y)
            if vidx is not None:
                self.drag_mode = "vertex"
                self.drag_vertex_index = vidx
                self.drag_start_world = (wx, wy)
                self.drag_origin_pos = self.board_outline.vertices[vidx]

                self.selected_vertex_index = vidx
                vx, vy = self.board_outline.vertices[vidx]
                self.var_vertex_idx.set(str(vidx))
                self.var_vx.set(f"{vx:.3f}")
                self.var_vy.set(f"{vy:.3f}")
                # self.var_status.set(f"拖动板框顶点 #{vidx}")
                return

        # 再尝试电极（拖动单个电极）
        hit_idx = self._hit_test_electrode(event.x, event.y)
        if hit_idx is None:
            # 点击空白：清空选择
            self.selected_electrode_indices = []
            self._populate_electrode_panel_from_selection()
            self.update_selection_info()
            self.redraw()
            return

        self.drag_mode = "electrode"
        self.drag_electrode_index = hit_idx
        self.drag_start_world = (wx, wy)
        elec = self.electrodes[hit_idx]
        self.drag_origin_pos = (elec.x, elec.y)

        # 处理选择（支持 Shift 多选）
        SHIFT_MASK = 0x0001
        if event.state & SHIFT_MASK:
            if hit_idx in self.selected_electrode_indices:
                self.selected_electrode_indices.remove(hit_idx)
            else:
                self.selected_electrode_indices.append(hit_idx)
        else:
            self.selected_electrode_indices = [hit_idx]

        self._populate_electrode_panel_from_selection()
        self.update_selection_info()
        self.redraw()

    def on_canvas_alt_press(self, event):
        """Alt + 左键：设置参考点 A。返回 'break' 阻止普通单击 handler 继续执行。"""
        self._set_ref_point_A_on_click(event)
        return "break"

    def on_canvas_drag(self, event):
        if self.drag_mode is None or self.drag_start_world is None:
            return
        wx, wy = self.canvas_to_world(event.x, event.y)
        sx, sy = self.drag_start_world
        dx = wx - sx
        dy = wy - sy

        # 更新对象位置
        if self.drag_mode == "electrode" and self.drag_electrode_index is not None:
            base_x, base_y = self.drag_origin_pos
            elec = self.electrodes[self.drag_electrode_index]
            elec.x = base_x + dx
            elec.y = base_y + dy
            self.var_x.set(f"{elec.x:.3f}")
            self.var_y.set(f"{elec.y:.3f}")
        elif self.drag_mode == "vertex" and self.drag_vertex_index is not None:
            base_x, base_y = self.drag_origin_pos
            vx = base_x + dx
            vy = base_y + dy
            self.board_outline.vertices[self.drag_vertex_index] = (vx, vy)
            self.var_vx.set(f"{vx:.3f}")
            self.var_vy.set(f"{vy:.3f}")

        # 重绘图形
        self.redraw()

        # 绘制拖动指示（十字 + 线 + 文本）
        self._draw_drag_indicator(wx, wy)

    def on_canvas_release(self, event):
        if self.drag_mode is not None:
            self.canvas.delete(self.drag_indicator_tag)
            if self.drag_mode == "electrode":
                pass
                # self.var_status.set("电极拖动完成")
            elif self.drag_mode == "vertex":
                # self.var_status.set("板框顶点拖动完成")
                pass
        self.drag_mode = None
        self.drag_electrode_index = None
        self.drag_vertex_index = None
        self.drag_start_world = None
        self.drag_origin_pos = None

    def on_canvas_double_click(self, event):
        # 仅用于结束多边形绘制
        if self.drawing_board_polygon:
            if len(self.temp_polygon_vertices) >= 3:
                self.board_outline = BoardOutline(self.temp_polygon_vertices[:])
                self.temp_polygon_vertices = []
                self.drawing_board_polygon = False
                self.var_status.set("多边形板框已创建")
                self.selected_vertex_index = None
                self.var_vertex_idx.set("-")
            else:
                messagebox.showwarning("提示", "多边形至少需要 3 个顶点")
            self.redraw()

    def on_canvas_motion(self, event):
        # 拖动时测量由 on_canvas_drag 负责，这里只管 Shift / Ctrl 测量
        if self.drag_mode is not None:
            return

        self.canvas.delete(self.measure_indicator_tag)

        state = event.state
        SHIFT_MASK = 0x0001
        CTRL_MASK = 0x0004

        wx, wy = self.canvas_to_world(event.x, event.y)

        # Shift：电极到最近板框距离
        if state & SHIFT_MASK:
            self._update_shift_measure(event, wx, wy)
            return

        # Ctrl：参考点 A 到另一个电极/顶点的距离和角度
        if state & CTRL_MASK and self.ref_point_A is not None:
            self._update_ctrl_measure(event, wx, wy)
            return

    # === 命中测试 ===

    def _hit_test_electrode(self, x_px, y_px):
        min_dist = 9999
        hit = None
        for idx, elec in enumerate(self.electrodes):
            cx, cy = self.world_to_canvas(elec.x, elec.y)
            d = math.hypot(cx - x_px, cy - y_px)
            if d < 12 and d < min_dist:
                min_dist = d
                hit = idx
        return hit

    def _hit_test_vertex(self, x_px, y_px):
        if not self.board_outline or not self.board_outline.vertices:
            return None
        for i, (vx, vy) in enumerate(self.board_outline.vertices):
            cx, cy = self.world_to_canvas(vx, vy)
            if abs(cx - x_px) <= 6 and abs(cy - y_px) <= 6:
                return i
        return None

    # === 拖动指示绘制 ===

    def _draw_drag_indicator(self, wx, wy):
        self.canvas.delete(self.drag_indicator_tag)
        if self.drag_start_world is None:
            return

        sx, sy = self.drag_start_world
        dx = wx - sx
        dy = wy - sy
        length = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        sx_px, sy_px = self.world_to_canvas(sx, sy)
        cx_px, cy_px = self.world_to_canvas(wx, wy)

        self.canvas.create_line(
            sx_px, sy_px, cx_px, cy_px,
            fill="magenta",
            width=2,
            dash=(3, 3),
            tags=self.drag_indicator_tag
        )

        size = 6
        self.canvas.create_line(
            cx_px - size, cy_px, cx_px + size, cy_px,
            fill="magenta",
            width=1,
            tags=self.drag_indicator_tag
        )
        self.canvas.create_line(
            cx_px, cy_px - size, cx_px, cy_px + size,
            fill="magenta",
            width=1,
            tags=self.drag_indicator_tag
        )

        text = f"{length:.2f} mm @ {angle_deg:.1f}°"
        self.canvas.create_text(
            cx_px + 15,
            cy_px - 15,
            text=text,
            anchor="w",
            fill="magenta",
            font=("Arial", 9, "bold"),
            tags=self.drag_indicator_tag
        )

    # === 参考点 A ===

    def _set_ref_point_A_on_click(self, event):
        wx, wy = self.canvas_to_world(event.x, event.y)

        hit_e = self._hit_test_electrode(event.x, event.y)
        label = "A"
        if hit_e is not None:
            elec = self.electrodes[hit_e]
            self.ref_point_A = (elec.x, elec.y)
            label = f"A:Elec#{elec.id}"
            self.var_status.set(f"参考点 A 设置为电极 #{elec.id}")
            self.selected_electrode_indices = [hit_e]
            self._populate_electrode_panel_from_selection()
            self.update_selection_info()
        else:
            hit_v = self._hit_test_vertex(event.x, event.y)
            if hit_v is not None and self.board_outline.vertices:
                vx, vy = self.board_outline.vertices[hit_v]
                self.ref_point_A = (vx, vy)
                label = f"A:V{hit_v}"
                self.var_status.set(f"参考点 A 设置为板框顶点 #{hit_v}")
                self.selected_vertex_index = hit_v
                self.var_vertex_idx.set(str(hit_v))
                self.var_vx.set(f"{vx:.3f}")
                self.var_vy.set(f"{vy:.3f}")
            else:
                self.ref_point_A = (wx, wy)
                self.var_status.set("参考点 A 已设置为当前坐标")

        self._draw_ref_point_A(label)

    def _draw_ref_point_A(self, label="A"):
        self.canvas.delete(self.ref_point_tag)
        if self.ref_point_A is None:
            return
        ax, ay = self.ref_point_A
        ax_px, ay_px = self.world_to_canvas(ax, ay)
        size = 6
        self.canvas.create_oval(
            ax_px - size,
            ay_px - size,
            ax_px + size,
            ay_px + size,
            outline="purple",
            width=2,
            tags=self.ref_point_tag
        )
        self.canvas.create_text(
            ax_px + 10,
            ay_px - 10,
            text=label,
            anchor="w",
            fill="purple",
            font=("Arial", 9, "bold"),
            tags=self.ref_point_tag
        )

    # === Shift 测量：电极到最近板框 ===

    def _update_shift_measure(self, event, wx, wy):
        if not self.board_outline or not self.board_outline.vertices:
            return
        hit_e = self._hit_test_electrode(event.x, event.y)
        if hit_e is None:
            return
        elec = self.electrodes[hit_e]
        px, py = elec.x, elec.y
        dist, (qx, qy) = self._point_to_polygon_distance(px, py, self.board_outline.vertices)
        if dist is None:
            return

        self.canvas.delete(self.measure_indicator_tag)

        ex_px, ex_py = self.world_to_canvas(px, py)
        qx_px, qy_px = self.world_to_canvas(qx, qy)

        self.canvas.create_line(
            ex_px, ex_py, qx_px, qy_px,
            fill="green",
            width=2,
            dash=(3, 3),
            tags=self.measure_indicator_tag
        )

        size = 5
        self.canvas.create_line(
            ex_px - size, ex_py, ex_px + size, ex_py,
            fill="green",
            width=1,
            tags=self.measure_indicator_tag
        )
        self.canvas.create_line(
            ex_px, ex_py - size, ex_px, ex_py + size,
            fill="green",
            width=1,
            tags=self.measure_indicator_tag
        )

        text = f"d = {dist:.2f} mm"
        self.canvas.create_text(
            ex_px + 12,
            ex_py - 12,
            text=text,
            anchor="w",
            fill="green",
            font=("Arial", 9, "bold"),
            tags=self.measure_indicator_tag
        )

    def _point_to_polygon_distance(self, px, py, vertices):
        if len(vertices) < 2:
            return None, (px, py)
        min_dist = None
        closest_pt = (px, py)
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            dist, cx, cy = self._point_to_segment_distance(px, py, x1, y1, x2, y2)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                closest_pt = (cx, cy)
        return min_dist, closest_pt

    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0:
            dist = math.hypot(px - x1, py - y1)
            return dist, x1, y1
        t = (wx * vx + wy * vy) / seg_len2
        t = max(0.0, min(1.0, t))
        cx = x1 + t * vx
        cy = y1 + t * vy
        dist = math.hypot(px - cx, py - cy)
        return dist, cx, cy

    # === Ctrl 测量：A → B ===

    def _update_ctrl_measure(self, event, wx, wy):
        hit_e = self._hit_test_electrode(event.x, event.y)
        B = None
        label = ""
        if hit_e is not None:
            elec = self.electrodes[hit_e]
            B = (elec.x, elec.y)
            label = f"Elec#{elec.id}"
        else:
            hit_v = self._hit_test_vertex(event.x, event.y)
            if hit_v is not None and self.board_outline.vertices:
                vx, vy = self.board_outline.vertices[hit_v]
                B = (vx, vy)
                label = f"V{hit_v}"
        if B is None or self.ref_point_A is None:
            return

        ax, ay = self.ref_point_A
        bx, by = B
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))

        self.canvas.delete(self.measure_indicator_tag)

        ax_px, ay_px = self.world_to_canvas(ax, ay)
        bx_px, by_px = self.world_to_canvas(bx, by)

        self.canvas.create_line(
            ax_px, ay_px, bx_px, by_px,
            fill="blue",
            width=2,
            dash=(4, 2),
            tags=self.measure_indicator_tag
        )

        size = 6
        self.canvas.create_line(
            bx_px - size, by_px, bx_px + size, by_px,
            fill="blue",
            width=1,
            tags=self.measure_indicator_tag
        )
        self.canvas.create_line(
            bx_px, by_px - size, bx_px, by_px + size,
            fill="blue",
            width=1,
            tags=self.measure_indicator_tag
        )

        text = f"A → {label}: {length:.2f} mm @ {angle_deg:.1f}°"
        self.canvas.create_text(
            bx_px + 15,
            by_px - 15,
            text=text,
            anchor="w",
            fill="blue",
            font=("Arial", 9, "bold"),
            tags=self.measure_indicator_tag
        )

    # === 电极属性面板 ===

    def update_selection_info(self):
        n = len(self.selected_electrode_indices)
        total = len(self.electrodes)
        self.var_sel_info.set(f"导联总数：{total}，当前选中：{n} 个电极")

    def _populate_electrode_panel_from_selection(self):
        if len(self.selected_electrode_indices) == 1:
            elec = self.electrodes[self.selected_electrode_indices[0]]
            self.var_x.set(f"{elec.x:.3f}")
            self.var_y.set(f"{elec.y:.3f}")
            self.var_d.set(f"{elec.diameter:.3f}")
            self.var_type.set(elec.etype)
            self.var_group.set(elec.group)
        elif len(self.selected_electrode_indices) == 0:
            self.var_x.set("")
            self.var_y.set("")
            self.var_d.set("")
            self.var_group.set("")
        else:
            self.var_x.set("")
            self.var_y.set("")
            self.var_d.set("")
        self.update_selection_info()

    def apply_electrode_properties(self):
        if not self.selected_electrode_indices:
            messagebox.showinfo("提示", "请先选择一个或多个电极")
            return
        try:
            x = self.var_x.get().strip()
            y = self.var_y.get().strip()
            d = self.var_d.get().strip()
            g = self.var_group.get().strip()
            etype = self.var_type.get().strip()

            for idx in self.selected_electrode_indices:
                elec = self.electrodes[idx]
                if x:
                    elec.x = float(x)
                if y:
                    elec.y = float(y)
                if d:
                    elec.diameter = max(0.1, float(d))
                if g:
                    elec.group = g
                if etype in ("measurement", "reference"):
                    elec.etype = etype

            self.var_status.set("已更新选中电极属性")
            self.redraw()
        except ValueError:
            messagebox.showerror("错误", "请输入合法的数字")

    # === 分组操作 ===

    def assign_group_to_selected(self):
        if not self.selected_electrode_indices:
            messagebox.showinfo("提示", "请先选择电极")
            return
        gname = self.var_group_name.get().strip()
        if not gname:
            messagebox.showerror("错误", "组名不能为空")
            return
        for idx in self.selected_electrode_indices:
            self.electrodes[idx].group = gname
        self.var_status.set(f"已将 {len(self.selected_electrode_indices)} 个电极设置为组 '{gname}'")
        self.redraw()

    def translate_group(self):
        gname = self.var_group_name.get().strip()
        if not gname:
            messagebox.showerror("错误", "请先输入要平移的组名")
            return
        try:
            dx = float(self.var_dx.get())
            dy = float(self.var_dy.get())
        except ValueError:
            messagebox.showerror("错误", "ΔX/ΔY 必须为数字")
            return
        count = 0
        for elec in self.electrodes:
            if elec.group == gname:
                elec.x += dx
                elec.y += dy
                count += 1
        self.var_status.set(f"组 '{gname}' 已平移 ({dx}, {dy}) mm，影响 {count} 个电极")
        self.redraw()

    # === 线性布局 ===

    def linear_layout_selected(self):
        if not self.selected_electrode_indices:
            messagebox.showinfo("提示", "请先选择至少 1 个电极")
            return
        try:
            x0 = float(self.var_lx.get())
            y0 = float(self.var_ly.get())
            sp = float(self.var_lsp.get())
        except ValueError:
            messagebox.showerror("错误", "起点和间距必须为数字")
            return
        direction = self.var_ldir.get()
        ordered = sorted(self.selected_electrode_indices)
        for i, idx in enumerate(ordered):
            elec = self.electrodes[idx]
            if direction == "horizontal":
                elec.x = x0 + i * sp
                elec.y = y0
            else:
                elec.x = x0
                elec.y = y0 + i * sp
        self.var_status.set(f"已对 {len(ordered)} 个电极应用线性布局")
        self.redraw()
        self._populate_electrode_panel_from_selection()

    # === 板框 ===

    def create_rect_board(self):
        try:
            w = float(
                simpledialog.askstring("矩形板框", "宽度 W (mm)：", initialvalue="80.0")
                or "0"
            )
            h = float(
                simpledialog.askstring("矩形板框", "高度 H (mm)：", initialvalue="80.0")
                or "0"
            )
        except ValueError:
            messagebox.showerror("错误", "宽度/高度必须为数字")
            return
        if w <= 0 or h <= 0:
            messagebox.showerror("错误", "宽度/高度必须为正数")
            return
        x0, y0 = 10.0, 10.0
        verts = [
            (x0, y0),
            (x0 + w, y0),
            (x0 + w, y0 + h),
            (x0, y0 + h),
        ]
        self.board_outline = BoardOutline(verts)
        self.drawing_board_polygon = False
        self.temp_polygon_vertices = []
        self.selected_vertex_index = None
        self.var_vertex_idx.set("-")
        self.var_status.set("矩形板框已创建")
        self.redraw()

    def start_draw_board_polygon(self):
        self.drawing_board_polygon = True
        self.temp_polygon_vertices = []
        self.selected_vertex_index = None
        self.var_vertex_idx.set("-")
        self.var_status.set("在画布上点击添加顶点，双击闭合多边形")
        self.canvas.delete(self.drag_indicator_tag)
        self.canvas.delete(self.measure_indicator_tag)
        self.redraw()

    def clear_board(self):
        self.board_outline = BoardOutline([])
        self.temp_polygon_vertices = []
        self.drawing_board_polygon = False
        self.selected_vertex_index = None
        self.var_vertex_idx.set("-")
        self.canvas.delete(self.drag_indicator_tag)
        self.canvas.delete(self.measure_indicator_tag)
        self.var_status.set("板框已清除")
        self.redraw()

    def apply_vertex_coord(self):
        if self.selected_vertex_index is None:
            messagebox.showinfo("提示", "请先在画布上点击一个板框顶点")
            return
        if not self.board_outline or not self.board_outline.vertices:
            messagebox.showerror("错误", "当前没有板框")
            return
        try:
            x = float(self.var_vx.get())
            y = float(self.var_vy.get())
        except ValueError:
            messagebox.showerror("错误", "顶点坐标必须为数字")
            return
        if not (0 <= self.selected_vertex_index < len(self.board_outline.vertices)):
            return
        self.board_outline.vertices[self.selected_vertex_index] = (x, y)
        self.var_status.set(
            f"顶点 #{self.selected_vertex_index} 已更新为 ({x:.3f}, {y:.3f}) mm"
        )
        self.redraw()

    # === 保存 / 加载 ===

    def save_to_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        data = {
            "channels": len(self.electrodes),
            "board_outline": self.board_outline.to_dict(),
            "electrodes": [e.to_dict() for e in self.electrodes],
            "units": "mm",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.var_status.set(f"已保存到 {path}")

    def load_from_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("错误", f"读取失败：{e}")
            return

        self.board_outline = BoardOutline.from_dict(data.get("board_outline", {}))
        self.electrodes = [
            Electrode.from_dict(ed) for ed in data.get("electrodes", [])
        ]
        self.channel_count = data.get("channels", len(self.electrodes))

        self.selected_electrode_indices = []
        self.temp_polygon_vertices = []
        self.drawing_board_polygon = False
        self.selected_vertex_index = None
        self.var_vertex_idx.set("-")
        self.drag_mode = None
        self.ref_point_A = None
        self.canvas.delete(self.drag_indicator_tag)
        self.canvas.delete(self.measure_indicator_tag)
        self.canvas.delete(self.ref_point_tag)

        self.update_selection_info()
        self.redraw()
        self.var_status.set(f"已从 {path} 载入设计（{len(self.electrodes)} 导联）")


def main():
    root = tk.Tk()
    app = ElectrodeDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
